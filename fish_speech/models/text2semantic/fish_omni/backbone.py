import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.types import Device

from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.expert_cache import (
    EXPERT_TENSORS,
    GIB,
    ExpertCache,
    ExpertOffloadConfig,
    ExpertStaging,
    cache_capacities,
    check_cuda_budget,
    expert_bytes,
)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import FP8_DTYPE
from fish_speech.models.text2semantic.fish_omni.layers import (
    FullAttention,
    GatedDeltaNet,
    LayerContext,
    ZeroCenteredRMSNorm,
)
from fish_speech.models.text2semantic.fish_omni.moe import (
    SparseMoeBlock,
    clamp_to_float16,
    dequantize_fp8_blocks,
)

CHECKPOINT_BACKBONE_PREFIX = "model.language_model."
CHECKPOINT_LM_HEAD_KEY = "lm_head.weight"
# Per-expert checkpoint tensors, named after the backbone prefix became "model.".
EXPERT_KEY_PATTERN = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)"
)
# Hugging Face's fused expert tensors (unquantized checkpoints).
FUSED_EXPERT_KEY_PATTERN = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)"
)
STACKED_EXPERT_PARAMETER_PATTERN = re.compile(
    r"(model\.layers\.\d+\.mlp\.experts)\.(w13|w2)_(weight|scale_inv)"
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: FishOmniTextConfig,
        layer_index: int,
        *,
        fp8: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "full_attention":
            self.self_attn = FullAttention(config, dtype=dtype, device=device)
        else:
            self.linear_attn = GatedDeltaNet(config, dtype=dtype, device=device)
        self.mlp = SparseMoeBlock(config, fp8=fp8, dtype=dtype, device=device)
        norm = dict(dtype=dtype, device=device)
        self.input_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps, **norm
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps, **norm
        )

    def forward(self, x: Tensor, context: LayerContext) -> Tensor:
        """x is the residual stream, possibly wider than the compute dtype."""
        compute_dtype = self.input_layernorm.weight.dtype
        mixer = (
            self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        )
        x = x + mixer(self.input_layernorm(x).to(compute_dtype), context)
        return x + self.mlp(self.post_attention_layernorm(x).to(compute_dtype))


class FishOmniTextModel(nn.Module):
    def __init__(
        self,
        config: FishOmniTextConfig,
        *,
        fp8: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, index, fp8=fp8, dtype=dtype, device=device)
            for index in range(config.num_hidden_layers)
        )
        self.norm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device
        )


class FishOmniBackbone(nn.Module):
    """Qwen3.5-MoE hybrid text backbone + lm_head; layers may span several devices."""

    def __init__(
        self,
        config: FishOmniTextConfig,
        *,
        fp8_experts: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        # The residual stream of this model exceeds the float16 range.
        self.residual_dtype = torch.float32 if dtype == torch.float16 else dtype
        self.model = FishOmniTextModel(
            config, fp8=fp8_experts, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.max_batch_size = 0
        self.max_seq_len = 0
        self.inverse_frequencies: dict[torch.device, Tensor] = {}
        self.expert_offload_enabled = False

    def materialize(
        self, devices: Sequence[Device], *, offload_experts: bool = False
    ) -> None:
        """Allocate a meta-device backbone over devices in contiguous layer ranges.

        Ranges are balanced by bytes, counting the embedding on the first device and
        the final norm + lm_head on the last.
        """
        devices = [torch.device(d) for d in devices]
        if offload_experts:
            if len(devices) != 1 or not all(
                layer.mlp.experts.fp8 for layer in self.model.layers
            ):
                raise ValueError("expert offload requires one device and FP8 experts")
            device = devices[0]
            for module in (self.model.embed_tokens, self.model.norm, self.lm_head):
                module.to_empty(device=device)
            for layer in self.model.layers:
                experts = layer.mlp.experts
                # Allocate each module's own tensors, never recursively moving
                # the experts onto CUDA even transiently during startup.
                for module in layer.modules():
                    module.to_empty(
                        device="cpu" if module is experts else device, recurse=False
                    )
            self.expert_offload_enabled = True
            self.inverse_frequencies.clear()
            logger.info(f"FishOmni placement: backbone on {device}, FP8 experts on CPU")
            return
        layer_bytes = [module_bytes(layer) for layer in self.model.layers]
        head = module_bytes(self.model.embed_tokens)
        tail = module_bytes(self.lm_head) + module_bytes(self.model.norm)
        per_device = (head + sum(layer_bytes) + tail) / len(devices)
        self.model.embed_tokens.to_empty(device=devices[0])
        self.model.norm.to_empty(device=devices[-1])
        self.lm_head.to_empty(device=devices[-1])
        placement = defaultdict(list)
        cumulative = head
        for index, (layer, size) in enumerate(zip(self.model.layers, layer_bytes)):
            middle = cumulative + size / 2
            device = devices[min(int(middle // per_device), len(devices) - 1)]
            layer.to_empty(device=device)
            placement[str(device)].append(index)
            cumulative += size
        self.inverse_frequencies.clear()
        logger.info(
            "FishOmni backbone placement: "
            + ", ".join(f"{d}: layers {v[0]}-{v[-1]}" for d, v in placement.items())
            + f"; embed on {devices[0]}, lm_head on {devices[-1]}"
        )

    def setup_expert_offload(self, config: ExpertOffloadConfig) -> None:
        """Attach caches after weights/tower load; reserve room for codec + runtime."""
        if not self.expert_offload_enabled:
            raise RuntimeError("materialize with offload_experts=True first")
        experts = [layer.mlp.experts for layer in self.model.layers]
        if any(e.expert_cache is not None for e in experts):
            raise RuntimeError("expert caches are already initialized")
        sources = [{name: getattr(e, name) for name in EXPERT_TENSORS} for e in experts]
        sizes = [expert_bytes(source) for source in sources]
        counts = [e.num_experts for e in experts]
        slots = cache_capacities(
            sizes, counts, self.config.num_experts_per_tok, config.cache_bytes
        )
        device = self.model.embed_tokens.weight.device
        if device.type != "cuda":
            raise ValueError("expert offload execution requires CUDA")
        actual_bytes = sum(size * count for size, count in zip(sizes, slots))
        check_cuda_budget(device, actual_bytes, config.reserve_bytes)
        staging = ExpertStaging(max(sizes), pin_memory=True)
        # Build all caches before publishing any; no partially configured model.
        caches = [
            ExpertCache(source, count, device, staging, max_rows=config.max_rows)
            for source, count in zip(sources, slots)
        ]
        for module, cache in zip(experts, caches):
            module.expert_cache = cache
        host_bytes = sum(module_bytes(e) for e in experts)
        logger.info(
            f"Drama 3 expert offload: CPU masters {host_bytes / GIB:.2f} GiB, "
            f"GPU cache {actual_bytes / GIB:.2f} GiB "
            f"(budget {config.cache_bytes / GIB:.2f} GiB), "
            f"slots/layer {min(slots)}..{max(slots)}, prefill rows <= {config.max_rows}, "
            f"pinned staging {staging.buffer.numel() / 2**20:.2f} MiB; "
            "blocking copies, CUDA graphs disabled"
        )

    def expert_cache_stats(self) -> dict[str, int]:
        caches = [layer.mlp.experts.expert_cache for layer in self.model.layers]
        return {
            name: sum(getattr(cache, name) for cache in caches if cache is not None)
            for name in ("hits", "misses", "transferred_bytes", "groups")
        }

    def setup_caches(self, max_batch_size: int, max_seq_len: int) -> None:
        assert max_seq_len <= self.config.max_position_embeddings
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if layer.layer_type == "full_attention":
                layer.self_attn.allocate_cache(
                    max_batch_size, max_seq_len, self.dtype, device
                )
            else:
                layer.linear_attn.allocate_cache(max_batch_size, self.dtype, device)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

    def embed(self, input_ids: Tensor) -> Tensor:
        embeddings = self.model.embed_tokens
        return embeddings(input_ids.to(embeddings.weight.device))

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        input_pos: Tensor | None = None,
        static_kv_len: int | None = None,
    ) -> Tensor:
        """Final-norm hidden states [B, T, H] of B rows at the T positions input_pos.

        With caches, rows at positions > 0 continue the cached sequence; without,
        only a whole sequence from position 0 can run. static_kv_len (CUDA-graph
        decode) attends that many cache rows without a host sync.
        """
        assert (input_ids is None) != (inputs_embeds is None)
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.to(self.residual_dtype)
        batch, length = hidden.shape[:2]
        if static_kv_len is not None:
            assert input_pos is not None and length == 1
            assert 0 < static_kv_len <= self.max_seq_len
            return self.run_layers(
                hidden,
                input_pos,
                kv_len=static_kv_len,
                continue_state=True,
                use_cache=True,
            )
        if input_pos is None:
            input_pos = torch.arange(length, device=hidden.device)
        assert input_pos.shape == (length,)
        start, end = int(input_pos[0]), int(input_pos[-1]) + 1
        assert start >= 0 and end - start == length, "input_pos must be contiguous"
        use_cache = self.max_seq_len > 0
        if use_cache:
            assert (
                batch <= self.max_batch_size and end <= self.max_seq_len
            ), "rows exceed the caches"
        else:
            assert start == 0, "continuing a sequence needs setup_caches first"
        return self.run_layers(
            hidden, input_pos, kv_len=end, continue_state=start > 0, use_cache=use_cache
        )

    def run_layers(
        self,
        hidden: Tensor,
        input_pos: Tensor,
        *,
        kv_len: int,
        continue_state: bool,
        use_cache: bool,
    ) -> Tensor:
        rotary_dim = self.config.rotary_dim
        contexts: dict[torch.device, LayerContext] = {}
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if device not in contexts:
                frequencies = self.inverse_frequencies.get(device)
                if frequencies is None:
                    exponents = torch.arange(
                        0, rotary_dim, 2, dtype=torch.float32, device=device
                    )
                    frequencies = 1.0 / (
                        self.config.rope_theta ** (exponents / rotary_dim)
                    )
                    self.inverse_frequencies[device] = frequencies
                positions = input_pos.to(device)
                angles = positions.float()[:, None] * frequencies[None, :]
                angles = torch.cat((angles, angles), dim=-1)
                contexts[device] = LayerContext(
                    input_pos=positions,
                    cos=angles.cos(),
                    sin=angles.sin(),
                    kv_len=kv_len,
                    continue_state=continue_state,
                    use_cache=use_cache,
                )
            hidden = layer(hidden.to(device), contexts[device])
        norm = self.model.norm
        return norm(hidden.to(norm.weight.device)).to(self.dtype)

    def compute_logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden.to(self.lm_head.weight.device))


def module_bytes(module: nn.Module) -> int:
    tensors = [*module.parameters(), *module.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors)


def backbone_name(checkpoint_name: str) -> str:
    """Map a checkpoint key onto a FishOmniBackbone parameter name."""
    if checkpoint_name == CHECKPOINT_LM_HEAD_KEY:
        return "lm_head.weight"
    if checkpoint_name.startswith(CHECKPOINT_BACKBONE_PREFIX):
        return "model." + checkpoint_name[len(CHECKPOINT_BACKBONE_PREFIX) :]
    if checkpoint_name.startswith("model."):
        return checkpoint_name
    raise ValueError(f"checkpoint key {checkpoint_name!r} is not a backbone weight")


@torch.no_grad()
def load_backbone_weights(
    model: FishOmniBackbone, weights: Iterable[tuple[str, Tensor]]
) -> None:
    """Copy checkpoint tensors into model, filling every parameter slot exactly once.

    Slots are (name, expert, part): stacked expert tensors are filled per expert and
    per gate / up / down projection, everything else whole.
    """
    for layer in model.model.layers:
        if layer.mlp.experts.expert_cache is not None:
            layer.mlp.experts.expert_cache.invalidate()
    targets = dict(model.named_parameters())
    targets.update(model.named_buffers())
    expected: set[tuple[str, int, str]] = set()
    for name in targets:
        match = STACKED_EXPERT_PARAMETER_PATTERN.fullmatch(name)
        if match is None:
            expected.add((name, -1, "whole"))
            continue
        experts = model.get_submodule(match.group(1))
        parts = ("gate", "up") if match.group(2) == "w13" else ("down",)
        for expert in range(experts.num_experts):
            expected.update((name, expert, part) for part in parts)
    loaded: set[tuple[str, int, str]] = set()
    # FP8 checkpoint experts loaded into dense experts: a weight or scale waits here
    # for its partner, then the pair is dequantized on the target's device.
    pending_fp8: dict[tuple[str, int, str], dict[str, Tensor]] = {}

    def fill(name: str, expert: int, part: str, target: Tensor, tensor: Tensor) -> None:
        slot = (name, expert, part)
        if slot not in expected:
            raise ValueError(f"no parameter slot {slot}")
        if slot in loaded:
            raise ValueError(f"parameter slot {slot} filled twice")
        if target.shape != tensor.shape:
            raise ValueError(
                f"{slot}: checkpoint shape {tuple(tensor.shape)} != "
                f"{tuple(target.shape)}"
            )
        if target.dtype == FP8_DTYPE and tensor.dtype != FP8_DTYPE:
            raise ValueError(f"{slot}: expected {FP8_DTYPE}, got {tensor.dtype}")
        if target.dtype == torch.float16 and tensor.dtype != torch.float16:
            tensor = clamp_to_float16(tensor, f"{name}[{expert}].{part}")
        target.copy_(tensor)
        loaded.add(slot)

    for checkpoint_name, tensor in weights:
        name = backbone_name(checkpoint_name)
        expert_match = EXPERT_KEY_PATTERN.fullmatch(name)
        fused_match = FUSED_EXPERT_KEY_PATTERN.fullmatch(name)
        if expert_match is not None:
            prefix = f"model.layers.{expert_match['layer']}.mlp.experts"
            experts = model.get_submodule(prefix)
            expert = int(expert_match["expert"])
            projection = expert_match["proj"]
            is_scale = expert_match["kind"] == "weight_scale_inv"
            if not is_scale and experts.fp8 and tensor.dtype != FP8_DTYPE:
                raise ValueError(f"{checkpoint_name}: expected an FP8 expert weight")
            if not experts.fp8 and (is_scale or tensor.dtype == FP8_DTYPE):
                key = (prefix, expert, projection)
                halves = pending_fp8.setdefault(key, {})
                if expert_match["kind"] in halves:
                    raise ValueError(f"{checkpoint_name} appears twice")
                halves[expert_match["kind"]] = tensor
                if len(halves) < 2:
                    continue
                del pending_fp8[key]
                target = experts.w13_weight
                tensor = dequantize_fp8_blocks(
                    halves["weight"].to(target.device),
                    halves["weight_scale_inv"].to(target.device),
                    torch.float32,
                )
                if target.dtype == torch.float16:
                    tensor = clamp_to_float16(tensor, checkpoint_name)
                tensor = tensor.to(target.dtype)
                is_scale = False
            suffix = "scale_inv" if is_scale else "weight"
            if projection == "down_proj":
                full_name = f"{prefix}.w2_{suffix}"
                fill(full_name, expert, "down", targets[full_name][expert], tensor)
            else:
                full_name = f"{prefix}.w13_{suffix}"
                rows = targets[full_name].shape[1] // 2
                gate = projection == "gate_proj"
                half = slice(0, rows) if gate else slice(rows, 2 * rows)
                fill(
                    full_name,
                    expert,
                    "gate" if gate else "up",
                    targets[full_name][expert, half],
                    tensor,
                )
        elif fused_match is not None:
            prefix = f"model.layers.{fused_match['layer']}.mlp.experts"
            if fused_match["proj"] == "down_proj":
                full_name = f"{prefix}.w2_weight"
                for expert, matrix in enumerate(tensor):
                    fill(full_name, expert, "down", targets[full_name][expert], matrix)
            else:
                full_name = f"{prefix}.w13_weight"
                rows = targets[full_name].shape[1] // 2
                for expert, matrix in enumerate(tensor):
                    target = targets[full_name][expert]
                    fill(full_name, expert, "gate", target[:rows], matrix[:rows])
                    fill(full_name, expert, "up", target[rows:], matrix[rows:])
        else:
            if name not in targets:
                raise ValueError(
                    f"checkpoint key {checkpoint_name!r} has no parameter {name!r}"
                )
            fill(name, -1, "whole", targets[name], tensor)

    if pending_fp8:
        raise ValueError(
            f"{len(pending_fp8)} FP8 experts lack a weight or scale, "
            f"e.g. {sorted(pending_fp8)[:4]}"
        )
    if missing := expected - loaded:
        raise ValueError(
            f"{len(missing)} backbone parameter slots were not loaded, "
            f"e.g. {sorted(missing)[:8]}"
        )
    logger.info(f"FishOmni backbone: loaded {len(loaded)} parameter slots")
