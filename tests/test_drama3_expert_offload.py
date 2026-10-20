"""CPU cache mechanics plus real CUDA parity tests (explicitly skipped without CUDA)."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fish_speech.models.text2semantic.fish_omni.backbone import (
    FishOmniBackbone,
    load_backbone_weights,
)
from fish_speech.models.text2semantic.fish_omni.config import (
    FishOmniConfig,
    FishOmniTextConfig,
)
from fish_speech.models.text2semantic.fish_omni.expert_cache import (
    EXPERT_TENSORS,
    GIB,
    ExpertCache,
    ExpertLRU,
    ExpertOffloadConfig,
    ExpertStaging,
    cache_capacities,
    check_cuda_budget,
    expert_bytes,
    row_groups,
)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    FP8_DTYPE,
    fp8_gemm_supported,
)
from fish_speech.models.text2semantic.fish_omni.generate import Drama3DecodeGraphs
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.moe import (
    MoEExperts,
    dequantize_fp8_blocks,
)


def tiny_config() -> FishOmniTextConfig:
    return FishOmniTextConfig(
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=32,
    )


def masters(experts: MoEExperts) -> dict[str, torch.Tensor]:
    return {name: getattr(experts, name) for name in EXPERT_TENSORS}


@torch.no_grad()
def fill_experts(experts: MoEExperts) -> None:
    for name in ("w13", "w2"):
        w = getattr(experts, f"{name}_weight")
        w.copy_((torch.randn(w.shape, device=w.device) * 0.2).to(FP8_DTYPE))
        getattr(experts, f"{name}_scale_inv").fill_(0.05)


def cpu_reference(x, ids, weights, w13, s13, w2, s2):
    """Exercises remapping, views and grouping, NOT the CUDA W8A8 kernel."""
    w13 = dequantize_fp8_blocks(w13, s13, x.dtype)
    w2 = dequantize_fp8_blocks(w2, s2, x.dtype)
    result = torch.zeros_like(x)
    for row in range(len(x)):
        for k, expert in enumerate(ids[row].tolist()):
            gate, up = F.linear(x[row], w13[expert]).chunk(2)
            result[row] += F.linear(F.silu(gate) * up, w2[expert]) * weights[row, k]
    return result


class ConfigTest(unittest.TestCase):
    def test_opt_in_and_original_default(self):
        self.assertIsNone(ExpertOffloadConfig.from_env({}))
        self.assertIsNone(
            ExpertOffloadConfig.from_env(
                {
                    "FISH_DRAMA3_EXPERT_OFFLOAD": "0",
                    "FISH_DRAMA3_EXPERT_CACHE_GIB": "not used",
                }
            )
        )
        self.assertEqual(
            ExpertOffloadConfig.from_env(
                {
                    "FISH_DRAMA3_EXPERT_OFFLOAD": "1",
                }
            ),
            ExpertOffloadConfig(),
        )

    def test_fractional_budget(self):
        config = ExpertOffloadConfig.from_env(
            {
                "FISH_DRAMA3_EXPERT_OFFLOAD": "1",
                "FISH_DRAMA3_EXPERT_CACHE_GIB": "12.5",
                "FISH_DRAMA3_EXPERT_RESERVE_GIB": "0",
                "FISH_DRAMA3_EXPERT_PREFILL_ROWS": "7",
            }
        )
        self.assertEqual(
            (config.cache_bytes, config.reserve_bytes, config.max_rows),
            (int(12.5 * GIB), 0, 7),
        )

    def test_invalid_values_fail_instead_of_silently_loading_full_model(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig.from_env({"FISH_DRAMA3_EXPERT_OFFLOAD": "true"})
        for name, values in {
            "FISH_DRAMA3_EXPERT_CACHE_GIB": ("nan", "inf", "-1", "0", "12GB"),
            "FISH_DRAMA3_EXPERT_RESERVE_GIB": ("nan", "inf", "-1"),
            "FISH_DRAMA3_EXPERT_PREFILL_ROWS": ("0", "-1", "2.5"),
        }.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    ExpertOffloadConfig.from_env(
                        {
                            "FISH_DRAMA3_EXPERT_OFFLOAD": "1",
                            name: value,
                        }
                    )

    def test_total_cache_budget_and_minimum_route(self):
        sizes, counts = [256, 512, 1024], [6, 8, 10]
        caps = cache_capacities(sizes, counts, 2, 9000)
        self.assertTrue(all(2 <= n <= maximum for n, maximum in zip(caps, counts)))
        self.assertLessEqual(sum(s * n for s, n in zip(sizes, caps)), 9000)
        with self.assertRaisesRegex(ValueError, "too small"):
            cache_capacities(sizes, counts, 2, 3000)

    def test_budget_counts_reusable_allocator_memory(self):
        with (
            patch("torch.cuda.mem_get_info", return_value=(10, 32)),
            patch("torch.cuda.memory_reserved", return_value=6),
            patch("torch.cuda.memory_allocated", return_value=4),
        ):
            check_cuda_budget(torch.device("cuda:0"), 8, 4)
            with self.assertRaisesRegex(ValueError, "headroom"):
                check_cuda_budget(torch.device("cuda:0"), 9, 4)


class LRUTest(unittest.TestCase):
    def test_required_hit_cannot_be_evicted_by_earlier_miss(self):
        lru = ExpertLRU(6, 2)
        loads = []
        load = lambda e, s: loads.append((e, s))
        lru.resolve([0, 1], load)
        # 0 is oldest but needed later in the same group; only 1 may be evicted.
        mapping, hits, misses = lru.resolve([2, 0], load)
        self.assertEqual((hits, misses), (1, 1))
        self.assertEqual(mapping, {2: 1, 0: 0})
        self.assertEqual(loads[-1], (2, 1))

    def test_lru_eviction_reload_and_duplicate_ids(self):
        lru = ExpertLRU(6, 3)
        load = lambda e, s: None
        lru.resolve([0, 1, 2], load)
        lru.resolve([0, 0], load)
        lru.resolve([3], load)
        self.assertNotIn(1, lru.slots)
        _, hits, misses = lru.resolve([1, 1], load)
        self.assertEqual((hits, misses), (0, 1))

    def test_partial_copy_failure_invalidates_all_mappings(self):
        lru = ExpertLRU(6, 2)
        lru.resolve([0, 1], lambda e, s: None)
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            lru.resolve(
                [2], lambda e, s: (_ for _ in ()).throw(RuntimeError("copy failed"))
            )
        self.assertFalse(lru.slots)
        _, hits, misses = lru.resolve([0], lambda e, s: None)
        self.assertEqual((hits, misses), (0, 1))

    def test_invalid_ids_or_oversized_group_do_not_mutate_cache(self):
        lru = ExpertLRU(3, 2)
        lru.resolve([1], lambda e, s: None)
        for ids in ([0, 1, 2], [-1], [3]):
            with self.assertRaises(ValueError):
                lru.resolve(ids, lambda e, s: self.fail("must not copy"))
            self.assertEqual(dict(lru.slots), {1: 0})


class GroupTest(unittest.TestCase):
    def test_union_limit_and_row_limit(self):
        rows = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 0]]
        groups = list(row_groups(rows, 3, 2))
        self.assertEqual(groups, [(0, 2, [0, 1, 2]), (2, 4, [0, 2, 3]), (4, 5, [0])])
        self.assertEqual([row for a, b, _ in groups for row in rows[a:b]], rows)

    def test_one_row_is_never_partially_evaluated(self):
        with self.assertRaisesRegex(ValueError, "one token"):
            list(row_groups([[0, 1, 2]], 2, 16))
        self.assertEqual(list(row_groups([], 2, 16)), [])


class CacheMechanicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.experts = MoEExperts(6, 128, 128, fp8=True, device="cpu")
        fill_experts(self.experts)
        source = masters(self.experts)
        self.cache = ExpertCache(
            source,
            3,
            torch.device("cpu"),
            ExpertStaging(expert_bytes(source), pin_memory=False),
            max_rows=2,
        )

    def test_packed_weights_scales_and_strides_survive_slot_reuse(self):
        for required in ([4, 1, 3], [5, 4, 0], [1, 2, 3], [4, 1, 3]):
            mapping, _, _ = self.cache.lru.resolve(required, self.cache._load)
            for expert, slot in mapping.items():
                for name, tensor in masters(self.experts).items():
                    self.assertTrue(
                        torch.equal(
                            self.cache.views[name][slot].view(torch.uint8),
                            tensor[expert].view(torch.uint8),
                        )
                    )
        for name, tensor in self.cache.views.items():
            self.assertEqual(
                tensor.stride(0) * tensor.element_size(), self.cache.slot_bytes
            )
        self.assertEqual(self.cache.allocated_bytes, 3 * self.cache.slot_bytes)

    def test_prefill_decode_and_repeated_roles_match_full_cpu_math(self):
        for ids in (
            [[5, 0], [0, 1], [2, 3], [3, 4], [5, 2], [1, 0]],
            [[0, 5]],
            [[0, 5]],
            [[1, 2]],
            [[5, 0]],
        ):
            ids = torch.tensor(ids)
            x = torch.randn(len(ids), 128)
            weights = torch.rand(ids.shape)
            expected = cpu_reference(x, ids, weights, *masters(self.experts).values())
            actual = self.cache.forward(x, ids, weights, cpu_reference)
            torch.testing.assert_close(actual, expected)
        self.assertGreater(self.cache.hits, 0)
        self.assertGreater(self.cache.misses, 3)

    def test_all_hit_does_not_transfer_again(self):
        x, ids, weights = torch.randn(1, 128), torch.tensor([[5, 1]]), torch.ones(1, 2)
        self.cache.forward(x, ids, weights, cpu_reference)
        before = self.cache.transferred_bytes
        self.cache.forward(x, ids, weights, cpu_reference)
        self.assertEqual(self.cache.transferred_bytes, before)
        self.assertEqual(self.cache.hits, 2)

    def test_kernel_failure_invalidates_cache_then_next_request_recovers(self):
        x, ids, weights = torch.randn(1, 128), torch.tensor([[5, 1]]), torch.ones(1, 2)

        def fail(*args):
            raise RuntimeError("kernel failure")

        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            self.cache.forward(x, ids, weights, fail)
        self.assertFalse(self.cache.lru.slots)
        actual = self.cache.forward(x, ids, weights, cpu_reference)
        torch.testing.assert_close(
            actual, cpu_reference(x, ids, weights, *masters(self.experts).values())
        )

    def test_invalid_later_row_fails_before_any_expert_load(self):
        with self.assertRaises(ValueError):
            self.cache.forward(
                torch.randn(2, 128),
                torch.tensor([[0, 1], [2, 6]]),
                torch.ones(2, 2),
                cpu_reference,
            )
        self.assertEqual(self.cache.transferred_bytes, 0)
        for ids in (torch.ones(1, 2), torch.tensor([0, 1])):
            with self.assertRaises(ValueError):
                self.cache.forward(
                    torch.randn(1, 128), ids, torch.ones(1, 2), cpu_reference
                )

    def test_moe_dispatch_uses_cache_and_detects_replaced_masters(self):
        self.experts.expert_cache = self.cache
        x, ids, weights = torch.randn(1, 128), torch.tensor([[1, 4]]), torch.ones(1, 2)
        with patch(
            "fish_speech.models.text2semantic.fish_omni.moe.fp8_moe_forward",
            side_effect=cpu_reference,
        ) as kernel:
            torch.testing.assert_close(
                self.experts(x, ids, weights),
                cpu_reference(x, ids, weights, *masters(self.experts).values()),
            )
            kernel.assert_called_once()
        self.experts.w13_weight = self.experts.w13_weight.clone()
        with self.assertRaisesRegex(RuntimeError, "masters were replaced"):
            self.experts(x, ids, weights)

    def test_cache_not_serialized_and_empty_input(self):
        self.experts.expert_cache = self.cache
        self.assertEqual(set(self.experts.state_dict()), set(EXPERT_TENSORS))
        got = self.cache.forward(
            torch.empty(0, 128),
            torch.empty(0, 2, dtype=torch.long),
            torch.empty(0, 2),
            lambda *a: self.fail("no kernel for empty input"),
        )
        self.assertEqual(got.shape, (0, 128))

    def test_loading_state_invalidates_cached_weight_values(self):
        self.experts.expert_cache = self.cache
        x, ids, weights = torch.randn(1, 128), torch.tensor([[1, 4]]), torch.ones(1, 2)
        self.cache.forward(x, ids, weights, cpu_reference)
        self.assertTrue(self.cache.lru.slots)
        state = {name: tensor.clone() for name, tensor in self.experts.state_dict().items()}
        state["w13_scale_inv"].mul_(0.5)
        self.experts.load_state_dict(state)
        self.assertFalse(self.cache.lru.slots)
        got = self.cache.forward(x, ids, weights, cpu_reference)
        torch.testing.assert_close(got, cpu_reference(x, ids, weights, *masters(self.experts).values()))


class PlacementTest(unittest.TestCase):
    @torch.no_grad()
    def test_original_per_expert_checkpoint_loads_directly_into_cpu_masters(self):
        torch.manual_seed(31)
        source = FishOmniBackbone(tiny_config(), fp8_experts=True, device="cpu")
        for p in source.parameters():
            p.normal_(0, 0.1)
        for layer in source.model.layers:
            fill_experts(layer.mlp.experts)
        checkpoint = []
        for name, tensor in source.state_dict().items():
            if ".experts." not in name:
                checkpoint.append((name, tensor))
                continue
            prefix, key = name.rsplit(".", 1)
            projection, kind = key.split("_", 1)
            suffix = "weight_scale_inv" if kind == "scale_inv" else "weight"
            for index, expert in enumerate(tensor):
                parts = (
                    [("down_proj", expert)]
                    if projection == "w2"
                    else list(zip(("gate_proj", "up_proj"), expert.chunk(2)))
                )
                checkpoint.extend(
                    (f"{prefix}.{index}.{proj}.{suffix}", value)
                    for proj, value in parts
                )
        with torch.device("meta"):
            target = FishOmniBackbone(tiny_config(), fp8_experts=True)
        target.materialize(["cpu"], offload_experts=True)
        load_backbone_weights(target, iter(checkpoint))
        for name, expected in source.state_dict().items():
            self.assertTrue(
                torch.equal(
                    target.state_dict()[name].view(torch.uint8),
                    expected.view(torch.uint8),
                ),
                name,
            )

    def test_invalid_offload_mode_rejected_before_any_model_allocation(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(json.dumps({}))
            for devices, tower, forced in (
                (["cuda:0", "cuda:1"], "cuda:1", "1"),
                (["cuda:0"], "cuda:1", "1"),
                (["cpu"], "cpu", "1"),
                (["cuda:0"], "cuda:0", "0"),
            ):
                with (
                    patch.dict(
                        "os.environ",
                        {
                            "FISH_DRAMA3_EXPERT_OFFLOAD": "1",
                            "FISH_DRAMA3_FP8_EXPERTS": forced,
                        },
                        clear=True,
                    ),
                    patch.object(
                        FishOmniConfig,
                        "from_dict",
                        return_value=SimpleNamespace(fp8_experts=True),
                    ),
                    patch(
                        "fish_speech.models.text2semantic.fish_omni.model.fp8_gemm_supported",
                        return_value=True,
                    ),
                    patch.object(FishOmniBackbone, "materialize") as allocate,
                    self.assertRaisesRegex(ValueError, "same single cuda:N"),
                ):
                    FishOmniModel.from_pretrained(
                        root,
                        devices=devices,
                        tower_device=tower,
                        dtype=torch.bfloat16,
                    )
                allocate.assert_not_called()

    def test_experts_allocated_on_cpu_without_materializing_full_device_layer(self):
        with torch.device("meta"):
            model = FishOmniBackbone(tiny_config(), fp8_experts=True)
        # Meta substitutes for the GPU: experts must become real CPU tensors,
        # while every non-expert tensor stays on the requested target device.
        model.materialize(["meta"], offload_experts=True)
        self.assertTrue(model.expert_offload_enabled)
        for name, tensor in list(model.named_parameters()) + list(
            model.named_buffers()
        ):
            expected = "cpu" if ".experts." in name else "meta"
            self.assertEqual(tensor.device.type, expected, name)

    def test_disabled_placement_remains_original(self):
        with torch.device("meta"):
            model = FishOmniBackbone(tiny_config(), fp8_experts=True)
        model.materialize(["cpu"])
        self.assertFalse(model.expert_offload_enabled)
        self.assertTrue(all(t.device.type == "cpu" for t in model.buffers()))
        self.assertEqual(
            model.expert_cache_stats(),
            dict(
                hits=0,
                misses=0,
                transferred_bytes=0,
                groups=0,
            ),
        )

    def test_rejects_dense_experts_and_multiple_devices(self):
        for fp8, devices in ((False, ["cpu"]), (True, ["cpu", "cpu"])):
            with torch.device("meta"):
                model = FishOmniBackbone(tiny_config(), fp8_experts=fp8)
            with self.assertRaises(ValueError):
                model.materialize(devices, offload_experts=True)

    def test_graph_guard_does_not_rely_on_parameter_device_detection(self):
        from types import SimpleNamespace

        model = SimpleNamespace(backbone=SimpleNamespace(expert_offload_enabled=True))
        with patch.dict("os.environ", {"FISH_DRAMA3_CUDA_GRAPH": "1"}):
            self.assertIsNone(Drama3DecodeGraphs.for_model(model, 100))


class VerificationToolTest(unittest.TestCase):
    def test_plan_uses_meta_only_and_includes_scales(self):
        from types import SimpleNamespace

        from tools.drama3_offload_verify import make_plan

        settings = ExpertOffloadConfig(cache_bytes=400000, reserve_bytes=0)
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text("{}")
            with (
                patch.object(FishOmniConfig, "from_dict", return_value=SimpleNamespace(
                    fp8_experts=True, text=tiny_config(),
                )),
                patch("torch.cuda.mem_get_info", side_effect=AssertionError("no CUDA in plan mode")),
            ):
                report = make_plan(Path(root), settings)
        self.assertEqual(report["mode"], "plan_only")
        self.assertGreater(report["cpu_expert_weight_bytes"], 6 * 2 * 3 * 128 * 128)
        self.assertLessEqual(report["gpu_expert_cache_bytes"], settings.cache_bytes)
        self.assertTrue(all(n >= 2 for n in report["slots_per_layer"]))

    def test_tool_refuses_accidental_full_gpu_fallback_and_saves_failure(self):
        from tools.drama3_offload_verify import main

        with tempfile.TemporaryDirectory() as root:
            argv = ["verify", "--checkpoint-path", "does-not-exist", "--run",
                    "--output-dir", root]
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("sys.argv", argv), redirect_stdout(StringIO()),
                patch("tools.drama3_offload_verify.make_plan") as plan,
            ):
                self.assertEqual(main(), 1)
            plan.assert_not_called()
            report = json.loads(Path(root, "report.json").read_text())
            self.assertFalse(report["smoke_pass"])
            self.assertIn("full-GPU fallback is disabled", report["error"])

    def test_run_rejects_cpu_instead_of_skipping_successfully(self):
        from tools.drama3_offload_verify import main

        with tempfile.TemporaryDirectory() as root:
            argv = ["verify", "--checkpoint-path", "unused", "--run",
                    "--device", "cpu", "--output-dir", root]
            with (
                patch.dict("os.environ", {"FISH_DRAMA3_EXPERT_OFFLOAD": "1"}, clear=True),
                patch("sys.argv", argv), redirect_stdout(StringIO()),
                patch("tools.drama3_offload_verify.make_plan", return_value={}),
            ):
                self.assertEqual(main(), 1)
            report = json.loads(Path(root, "report.json").read_text())
            self.assertIn("requires an available explicit CUDA device", report["error"])
            self.assertFalse(report["smoke_pass"])


@unittest.skipUnless(
    fp8_gemm_supported(["cuda:0"]), "requires actual SM89+ CUDA/Triton"
)
class CudaOffloadTest(unittest.TestCase):
    @torch.no_grad()
    def test_native_kernel_prefill_decode_eviction_and_stream_switch(self):
        torch.manual_seed(19)
        original = MoEExperts(6, 128, 256, fp8=True, device="cuda:0")
        fill_experts(original)
        offloaded = MoEExperts(6, 128, 256, fp8=True, device="cpu")
        offloaded.load_state_dict(original.state_dict())
        source = masters(offloaded)
        offloaded.expert_cache = ExpertCache(
            source,
            3,
            torch.device("cuda:0"),
            ExpertStaging(expert_bytes(source), pin_memory=True),
            max_rows=16,
        )
        for rows in (1, 27, 1, 300, 1):
            x = torch.randn(rows, 128, device="cuda:0", dtype=torch.bfloat16)
            weights, ids = torch.rand(rows, 6, device="cuda:0").topk(2, dim=-1)
            weights = weights.bfloat16()
            expected = original(x, ids, weights)
            got = offloaded(x, ids, weights)
            torch.testing.assert_close(got, expected, rtol=0.02, atol=2e-5)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            got = offloaded(x, ids, weights)
        stream.synchronize()
        torch.testing.assert_close(got, expected, rtol=0.02, atol=2e-5)
        self.assertTrue(all(t.device.type == "cpu" for t in offloaded.buffers()))

    @torch.no_grad()
    def test_backbone_teacher_forced_logits_and_state_reset(self):
        torch.manual_seed(23)
        original = FishOmniBackbone(
            tiny_config(), fp8_experts=True, dtype=torch.bfloat16, device="cuda:0"
        ).eval()
        for parameter in original.parameters():
            parameter.normal_(0, 0.05)
        for layer in original.model.layers:
            fill_experts(layer.mlp.experts)
        with torch.device("meta"):
            offloaded = FishOmniBackbone(tiny_config(), fp8_experts=True)
        offloaded.materialize(["cuda:0"], offload_experts=True)
        offloaded.load_state_dict(original.state_dict())
        size = expert_bytes(masters(offloaded.model.layers[0].mlp.experts))
        offloaded.setup_expert_offload(
            ExpertOffloadConfig(
                cache_bytes=size * 3 * 2,
                reserve_bytes=0,
                max_rows=16,
            )
        )
        for model in (original, offloaded):
            model.setup_caches(1, 64)
        tokens = torch.randint(0, 64, (1, 40), device="cuda:0")
        for a, b in ((0, 23), (23, 24), (24, 40), (0, 5), (5, 6)):
            pos = torch.arange(a, b, device="cuda:0")
            reference = original.compute_logits(original(tokens[:, a:b], input_pos=pos))
            actual = offloaded.compute_logits(offloaded(tokens[:, a:b], input_pos=pos))
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.002)


if __name__ == "__main__":
    unittest.main()
