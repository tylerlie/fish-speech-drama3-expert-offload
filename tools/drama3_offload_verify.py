"""Plan without weights, or explicitly run a local Drama 3 offload smoke test.

From the repository root: python -m tools.drama3_offload_verify --help
No downloading, dependency installation, or server lifecycle changes occur here.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fish_speech.models.text2semantic.fish_omni.backbone import (
    FishOmniBackbone,
    module_bytes,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.expert_cache import (
    EXPERT_TENSORS,
    GIB,
    ExpertOffloadConfig,
    cache_capacities,
    expert_bytes,
)


def make_plan(checkpoint: Path, settings: ExpertOffloadConfig) -> dict:
    config = FishOmniConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text())
    )
    if not config.fp8_experts:
        raise ValueError(
            "expert offload verification requires the original FP8 checkpoint"
        )
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True)
    layers = [layer.mlp.experts for layer in backbone.model.layers]
    sizes = [expert_bytes({n: getattr(e, n) for n in EXPERT_TENSORS}) for e in layers]
    slots = cache_capacities(
        sizes,
        [e.num_experts for e in layers],
        config.text.num_experts_per_tok,
        settings.cache_bytes,
    )
    cpu_bytes = sum(module_bytes(e) for e in layers)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "mode": "plan_only",
        "backbone_non_expert_weight_bytes": module_bytes(backbone) - cpu_bytes,
        "cpu_expert_weight_bytes": cpu_bytes,
        "gpu_expert_cache_bytes": sum(s * n for s, n in zip(sizes, slots)),
        "slots_per_layer": slots,
        "pinned_staging_bytes": max(sizes),
        "reserve_bytes": settings.reserve_bytes,
        "max_moe_rows_per_group": settings.max_rows,
        "note": "Weight geometry only; excludes tower/codec, runtime, allocator and driver. "
        "This does not establish that the full service fits 32GB.",
    }


@torch.inference_mode()
def run_check(args: argparse.Namespace, report: dict) -> bool:
    # Import the normal inference path only for an explicitly requested GPU run.
    import soundfile as sf

    from fish_speech.models.text2semantic.fish_omni.generate import (
        build_drama3_prompt,
        generate_drama3,
    )
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        encode_audio,
        init_model,
        load_codec_model,
        resolve_codec_path,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ValueError(
            "--run requires an available explicit CUDA device, e.g. cuda:0"
        )
    if bool(args.reference_audio) != bool(args.reference_text):
        raise ValueError("reference audio and its transcript must be supplied together")
    if args.reference_audio and not args.reference_audio.is_file():
        raise FileNotFoundError(args.reference_audio)
    codec_path = resolve_codec_path(args.checkpoint_path, args.codec_path)
    props = torch.cuda.get_device_properties(device)
    report.update(
        mode="gpu_smoke_test",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=props.name,
        gpu_total_bytes=props.total_memory,
        runs=[],
        content_and_voice_quality="not_evaluated",
        measurement_scope="Local generation+codec decode; excludes server, upload, "
        "reference encoding and startup from per-run RTF. "
        "Memory peaks are PyTorch allocator peaks, not NVML process peaks.",
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, _ = init_model(args.checkpoint_path, args.device, torch.bfloat16)
    codec = load_codec_model(codec_path, args.device, model.dtype)
    torch.cuda.synchronize(device)
    report["load_seconds"] = time.perf_counter() - started
    report["startup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["startup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if not model.backbone.expert_offload_enabled:
        raise RuntimeError("refusing smoke test: expert offload did not activate")
    refs, transcripts = [], []
    started = time.perf_counter()
    if args.reference_audio:
        refs = [encode_audio(args.reference_audio, codec, args.device).cpu()]
        transcripts = [args.reference_text]
    torch.cuda.synchronize(device)
    report["reference_encoding_seconds"] = time.perf_counter() - started
    report["reference_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["reference_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    prompt = build_drama3_prompt(
        args.text, model.tokenizer, model.config, transcripts, refs
    )
    overall = True
    for index in range(args.repeat):
        torch.manual_seed(args.seed)
        before = model.backbone.expert_cache_stats()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        frames, reason = generate_drama3(
            model, prompt, max_new_tokens=args.max_new_tokens
        )
        if not len(frames):
            raise RuntimeError(
                "model emitted EOS without any audio; not a successful test"
            )
        audio = decode_to_audio(frames[:, 1:].T.contiguous().to(device), codec)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        # Read peaks BEFORE copying audio to CPU and writing the file.
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        audio = audio.detach().float().cpu().reshape(-1)
        finite = bool(torch.isfinite(audio).all())
        rms = float(audio.double().square().mean().sqrt())
        duration = len(audio) / codec.sample_rate
        after = model.backbone.expert_cache_stats()
        healthy = finite and math.isfinite(rms) and rms > 1e-6
        memory_ok = max(peak_allocated, peak_reserved) <= args.max_peak_gib * GIB
        passed = healthy and memory_ok and reason == "eos"
        path = args.output_dir / f"sample_{index + 1:02d}.wav"
        if finite:
            sf.write(path, audio.numpy(), codec.sample_rate, subtype="PCM_16")
        report["runs"].append(
            {
                "index": index + 1,
                "seed": args.seed,
                "finish": reason,
                "frames": len(frames),
                "duration_seconds": duration,
                "generation_and_decode_seconds": elapsed,
                "rtf": elapsed / duration,
                "finite": finite,
                "rms_dbfs": 20 * math.log10(rms) if rms > 0 and finite else None,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "cache_delta": {
                    key: after[key] - value for key, value in before.items()
                },
                "wav": str(path.resolve()) if finite else None,
                "smoke_pass": passed,
            }
        )
        overall &= passed
    startup_peak = max(
        report["startup_peak_allocated_bytes"],
        report["startup_peak_reserved_bytes"],
        report["reference_peak_allocated_bytes"],
        report["reference_peak_reserved_bytes"],
    )
    report["smoke_pass"] = overall and startup_peak <= args.max_peak_gib * GIB
    report["note"] = (
        "Smoke gates: nonempty finite audio, RMS > -120dBFS, EOS, and PyTorch peaks "
        f"<= {args.max_peak_gib} GiB. This does not prove text completeness, cloning "
        "quality, multi-role workflows, NVML memory peaks, or 5090 compatibility "
        "when tested on another GPU. Inspect WAVs and run the separate CUDA parity tests."
    )
    return report["smoke_pass"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--run", action="store_true", help="actually load model and generate on CUDA"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codec-path", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument(
        "--text", default="你好，这是一段声音生成测试，请完整读完这句话。"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-peak-gib", type=float, default=29.0)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/offload-check"))
    args = parser.parse_args()
    if (
        args.repeat < 1
        or args.max_new_tokens < 1
        or not args.text.strip()
        or not math.isfinite(args.max_peak_gib)
        or args.max_peak_gib <= 0
    ):
        parser.error("repeat/tokens/peak must be positive and text must be nonempty")
    report = {"created_at": datetime.now(timezone.utc).isoformat()}
    try:
        settings = ExpertOffloadConfig.from_env()
        if settings is None:
            raise ValueError(
                "set FISH_DRAMA3_EXPERT_OFFLOAD=1 first; full-GPU fallback is disabled here"
            )
        report.update(make_plan(args.checkpoint_path, settings))
        if not args.run:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        passed = run_check(args, report)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", smoke_pass=False)
        passed = False
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    # Always retain diagnostics for explicitly requested GPU runs, including OOM.
    if args.run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
