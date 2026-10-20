"""Bounded, synchronous FP8 expert offload. No model math or sampling lives here."""

import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import torch
from torch import Tensor

GIB = 2**30
EXPERT_TENSORS = (
    "w13_weight",
    "w13_scale_inv",
    "w2_weight",
    "w2_scale_inv",
)


@dataclass(frozen=True)
class ExpertOffloadConfig:
    cache_bytes: int = 12 * GIB
    reserve_bytes: int = 4 * GIB
    max_rows: int = 16

    def __post_init__(self) -> None:
        if self.cache_bytes <= 0 or self.reserve_bytes < 0 or self.max_rows < 1:
            raise ValueError("expert cache must be positive, reserve >= 0, rows >= 1")

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "ExpertOffloadConfig | None":
        env = os.environ if env is None else env
        enabled = env.get("FISH_DRAMA3_EXPERT_OFFLOAD", "0").strip()
        if enabled not in ("0", "1"):
            raise ValueError("FISH_DRAMA3_EXPERT_OFFLOAD must be 0 or 1")
        if enabled == "0":
            return None

        def gib(name: str, default: str, *, positive: bool) -> int:
            try:
                value = Decimal(env.get(name, default).strip())
                if not value.is_finite() or value < 0 or (positive and value == 0):
                    raise ValueError
                result = int(value * GIB)
                if positive and result < 1:
                    raise ValueError
                return result
            except (InvalidOperation, ValueError, OverflowError):
                raise ValueError(
                    f"{name} must be a finite GiB value "
                    f"{'greater than' if positive else '>='} 0"
                ) from None

        name = "FISH_DRAMA3_EXPERT_PREFILL_ROWS"
        try:
            rows = int(env.get(name, "16"))
        except ValueError:
            raise ValueError(f"{name} must be a positive integer") from None
        return cls(
            cache_bytes=gib("FISH_DRAMA3_EXPERT_CACHE_GIB", "12", positive=True),
            reserve_bytes=gib("FISH_DRAMA3_EXPERT_RESERVE_GIB", "4", positive=False),
            max_rows=rows,
        )


def expert_bytes(sources: Mapping[str, Tensor]) -> int:
    """Size of ONE expert, including scales, aligned for packed slot views."""
    offset = 0
    for name in EXPERT_TENSORS:
        tensor = sources[name]
        size = tensor.element_size()
        offset = (offset + 255) // 256 * 256
        offset += math.prod(tensor.shape[1:]) * size
    return (offset + 255) // 256 * 256


def cache_capacities(
    sizes: Sequence[int], counts: Sequence[int], top_k: int, budget: int
) -> list[int]:
    """Equal byte quotas per layer; never exceed the total requested budget."""
    if not sizes or len(sizes) != len(counts) or top_k < 1:
        raise ValueError("invalid expert cache geometry")
    quota = budget // len(sizes)
    result = []
    for size, count in zip(sizes, counts):
        if size < 1 or count < top_k:
            raise ValueError("invalid expert count or size")
        capacity = min(count, quota // size)
        if capacity < top_k:
            raise ValueError(
                f"expert cache budget too small: each layer needs at least "
                f"{top_k} slots; this layer has {capacity}. Increase "
                "FISH_DRAMA3_EXPERT_CACHE_GIB"
            )
        result.append(capacity)
    return result


def row_groups(
    rows: Sequence[Sequence[int]], capacity: int, max_rows: int
) -> Iterator[tuple[int, int, list[int]]]:
    """Contiguous rows whose expert union fits; never split a row's top-k sum."""
    if capacity < 1 or max_rows < 1:
        raise ValueError("capacity and max_rows must be positive")
    start, required = 0, set()
    for index, row in enumerate(rows):
        unique = set(row)
        if len(unique) > capacity:
            raise ValueError("one token needs more experts than the cache capacity")
        if index > start and (
            len(required | unique) > capacity or index - start >= max_rows
        ):
            yield start, index, sorted(required)
            start, required = index, set()
        required.update(unique)
    if start < len(rows):
        yield start, len(rows), sorted(required)


class ExpertLRU:
    """Layer-local slots. All experts required by a group are protected together."""

    def __init__(self, count: int, capacity: int) -> None:
        if not 0 < capacity <= count:
            raise ValueError("invalid cache capacity")
        self.count = count
        self.capacity = capacity
        self.slots: OrderedDict[int, int] = OrderedDict()

    def clear(self) -> None:
        self.slots.clear()

    def resolve(
        self, required: Sequence[int], load: Callable[[int, int], None]
    ) -> tuple[dict[int, int], int, int]:
        required = list(dict.fromkeys(required))
        if len(required) > self.capacity or any(
            expert < 0 or expert >= self.count for expert in required
        ):
            raise ValueError("required experts exceed cache or model bounds")
        protected = set(required)
        hits = sum(expert in self.slots for expert in required)
        try:
            for expert in required:
                if expert in self.slots:
                    self.slots.move_to_end(expert)
                    continue
                if len(self.slots) < self.capacity:
                    used = set(self.slots.values())
                    slot = next(i for i in range(self.capacity) if i not in used)
                else:
                    victim = next(e for e in self.slots if e not in protected)
                    slot = self.slots.pop(victim)
                # Publish the mapping only after the complete weight+scale copy.
                load(expert, slot)
                self.slots[expert] = slot
        except BaseException:
            # A failed/partial copy must never leave a falsely valid slot behind.
            self.clear()
            raise
        return {e: self.slots[e] for e in required}, hits, len(required) - hits


class ExpertStaging:
    """One pinned expert buffer shared by all layers, not a pinned model copy."""

    def __init__(self, size: int, *, pin_memory: bool) -> None:
        self.buffer = torch.empty(
            size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
        ).zero_()
        self.lock = threading.Lock()


class ExpertCache:
    """CPU masters + fixed GPU byte slots, viewed by the original FP8 kernel.

    CUDA copies are blocking and use the caller's compute stream. An overwrite is
    ordered AFTER previous kernels using that slot; pinned staging is reusable
    when copy_ returns. Packed slots preserve original bytes and FP32 scales.
    The original kernel accepts explicit expert strides, so no new CUDA math is
    needed. CPU destination support exists only for unit tests of cache mechanics.
    """

    def __init__(
        self,
        sources: Mapping[str, Tensor],
        capacity: int,
        device: torch.device,
        staging: ExpertStaging,
        *,
        max_rows: int = 16,
    ) -> None:
        self.sources = {name: sources[name] for name in EXPERT_TENSORS}
        first = self.sources[EXPERT_TENSORS[0]]
        self.count = first.shape[0]
        for tensor in self.sources.values():
            if (
                tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or tensor.shape[0] != self.count
            ):
                raise ValueError("expert masters must be contiguous CPU tensors")
        self.device = torch.device(device)
        self.slot_bytes = expert_bytes(self.sources)
        if staging.buffer.numel() < self.slot_bytes or max_rows < 1:
            raise ValueError("invalid expert staging buffer or max_rows")
        self.lru = ExpertLRU(self.count, capacity)
        self.staging = staging
        self.max_rows = max_rows
        self.storage = torch.empty(
            (capacity, self.slot_bytes), dtype=torch.uint8, device=self.device
        )
        self.views: dict[str, Tensor] = {}
        self.ranges: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, source in self.sources.items():
            size = source.element_size()
            offset = (offset + 255) // 256 * 256
            end = offset + source[0].numel() * size
            self.ranges[name] = offset, end
            self.views[name] = (
                self.storage[:, offset:end]
                .view(source.dtype)
                .view(capacity, *source.shape[1:])
            )
            offset = end
        self.hits = self.misses = self.transferred_bytes = self.groups = 0
        self._stream_id: int | None = None

    @property
    def allocated_bytes(self) -> int:
        return self.storage.numel()

    def invalidate(self) -> None:
        """Forget cached copies before replacing any CPU master values."""
        with self.staging.lock:
            self.lru.clear()

    def _load(self, expert: int, slot: int) -> None:
        for name, source in self.sources.items():
            start, end = self.ranges[name]
            self.staging.buffer[start:end].copy_(
                source[expert].view(torch.uint8).reshape(-1)
            )
        self.storage[slot].copy_(
            self.staging.buffer[: self.slot_bytes], non_blocking=False
        )
        self.transferred_bytes += self.slot_bytes

    def forward(
        self, x: Tensor, ids: Tensor, weights: Tensor, kernel: Callable
    ) -> Tensor:
        if (
            x.device != self.device
            or ids.device != x.device
            or weights.device != x.device
        ):
            raise ValueError("offloaded experts and routing must use the cache device")
        if (
            x.ndim != 2
            or ids.ndim != 2
            or ids.shape != weights.shape
            or ids.shape[0] != x.shape[0]
            or ids.shape[1] < 1
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid expert routing shape or dtype")
        if not x.shape[0]:
            return torch.empty_like(x)
        with self.staging.lock:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "expert offload cannot run inside a CUDA graph"
                        )
                    stream = torch.cuda.current_stream(self.device).cuda_stream
                if self._stream_id is not None and stream != self._stream_id:
                    # Serialized calls may change streams; drain previous users.
                    torch.cuda.synchronize(self.device)
                self._stream_id = stream
            rows = ids.detach().cpu().tolist()
            if any(e < 0 or e >= self.count for row in rows for e in row):
                raise ValueError("expert ID outside the checkpoint's expert range")
            if any(len(set(row)) > self.lru.capacity for row in rows):
                raise ValueError("one token needs more experts than the cache capacity")
            result = torch.empty_like(x)
            try:
                for start, end, required in row_groups(
                    rows, self.lru.capacity, self.max_rows
                ):
                    mapping, hits, misses = self.lru.resolve(required, self._load)
                    self.hits += hits
                    self.misses += misses
                    self.groups += 1
                    remapped = torch.tensor(
                        [[mapping[e] for e in row] for row in rows[start:end]],
                        dtype=ids.dtype,
                        device=self.device,
                    )
                    result[start:end].copy_(
                        kernel(
                            x[start:end],
                            remapped,
                            weights[start:end],
                            *(self.views[name] for name in EXPERT_TENSORS),
                        )
                    )
            except BaseException:
                self.lru.clear()
                raise
            return result


def check_cuda_budget(
    device: torch.device, cache_bytes: int, reserve_bytes: int
) -> None:
    """Reserve is a startup headroom check, not a promise of peak runtime usage."""
    free, _ = torch.cuda.mem_get_info(device)
    reusable = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    if cache_bytes + reserve_bytes > free + reusable:
        raise ValueError(
            f"expert cache needs {cache_bytes / GIB:.2f} GiB + "
            f"{reserve_bytes / GIB:.2f} GiB headroom; only "
            f"{(free + reusable) / GIB:.2f} GiB available. Reduce "
            "FISH_DRAMA3_EXPERT_CACHE_GIB; do not remove runtime headroom blindly"
        )
