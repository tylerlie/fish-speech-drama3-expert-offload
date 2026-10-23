"""CPU-only ComfyUI client for the companion index_api.py adapter."""
import base64
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .audio import available_reference_media, validate_reference, encode_reference, encode_audio_input, decode_audio

UNCONNECTED = object()
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
PARAMETERS = {
    "emotion_mode": (["reference", "text", "vector", "audio"], {"default": "reference", "tooltip": "reference=跟随音色参考；text=情绪文字；vector=八维滑块；audio=独立情绪参考。"}),
    "emotion_text": ("STRING", {"default": "", "multiline": True, "tooltip": "仅 text 模式生效；留空时从生成文稿分析情绪。文稿不要写 Fish 专用标签。"}),
    "emo_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
    **{name: ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}) for name in EMOTIONS},
    "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 2.0, "step": 0.05}),
    "top_p": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 1.0, "step": 0.05}),
    "top_k": ("INT", {"default": 30, "min": 0, "max": 1000}),
    "num_beams": ("INT", {"default": 3, "min": 1, "max": 10}),
    "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 20.0, "step": 0.1}),
    "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
    "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 4096}),
    "max_text_tokens_per_segment": ("INT", {"default": 120, "min": 20, "max": 200}),
    "interval_silence": ("INT", {"default": 200, "min": 0, "max": 2000, "tooltip": "模型内部文本分段之间的静音，单位毫秒。"}),
    "use_random": ("BOOLEAN", {"default": False}),
    "use_memory_cache": ("BOOLEAN", {"default": False, "tooltip": "测速默认关闭参考特征复用；模型本身始终常驻。"}),
    "lang": (["ZH", "EN", "JA", "ES", "AR"], {"default": "ZH", "tooltip": "2.0 仅选 ZH/EN；2.5 支持这五种语言。"}),
    "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "仅 2.5 支持非 1 值；越大音频越长。"}),
}


class Reference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"audio": (available_reference_media(), {"default": ""}),
                             "text": ("STRING", {"default": "", "multiline": True}),
                             "audio_in": ("AUDIO",)}}
    RETURN_TYPES = ("INDEX_REFERENCE",)
    RETURN_NAMES = ("reference",)
    FUNCTION = "bind"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "上传或选取参考音频，亦可连接上游 AUDIO（优先）。不需要参考原文，备注不参与推理。音频可空。"

    @classmethod
    def VALIDATE_INPUTS(cls, audio="", text="", audio_in=UNCONNECTED, input_types=None):
        try:
            for name, kind in (input_types or {}).items():
                if kind != {"audio": "STRING", "text": "STRING", "audio_in": "AUDIO"}.get(name):
                    return "输入类型不匹配。"
            if "audio_in" in (input_types or {}) or audio_in is not UNCONNECTED or "audio" in (input_types or {}):
                return True
            validate_reference(audio)
            return True
        except ValueError as exc:
            return str(exc)

    @classmethod
    def IS_CHANGED(cls, audio="", text="", audio_in=UNCONNECTED):
        if audio_in is not UNCONNECTED:
            return float("nan")
        path = validate_reference(audio)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else "empty"

    def bind(self, audio="", text="", audio_in=UNCONNECTED):
        if audio_in is not UNCONNECTED:
            return ({"audio": encode_audio_input(audio_in)},)
        path = validate_reference(audio)
        return ({"audio": encode_reference(path)} if path else None,)


class Parameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": PARAMETERS}
    RETURN_TYPES = ("INDEX_PARAMETERS",)
    FUNCTION = "build"
    CATEGORY = "IndexTTS Local"

    def build(self, **values):
        result = {key: options[1]["default"] for key, options in PARAMETERS.items()}
        result.update(values)
        result["emotion_vector"] = [result.pop(name) for name in EMOTIONS]
        return (result,)


class Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True}),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "control_after_generate": True})},
                "optional": {"reference": ("INDEX_REFERENCE",), "emotion_reference": ("INDEX_REFERENCE",),
                             "parameters": ("INDEX_PARAMETERS",)}}
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "调用独立 Index API，兼容 2.0/2.5。音色参考留空使用服务器默认音色，不是真正无参考生成。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, text, seed=42, reference=None, emotion_reference=None, parameters=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("生成文稿不能为空。")
        payload = {**(parameters or Parameters().build()[0]), "text": text, "seed": seed}
        for key, ref in (("reference_audio", reference), ("emotion_audio", emotion_reference)):
            if ref is not None:
                if not isinstance(ref, dict) or not isinstance(ref.get("audio"), bytes):
                    raise ValueError("请连接 IndexTTS Reference 节点。")
                payload[key] = base64.b64encode(ref["audio"]).decode("ascii")
        if payload["emotion_mode"] != "audio":
            payload.pop("emotion_audio", None)
        elif not payload.get("emotion_audio"):
            raise ValueError("audio 情绪模式需要填写独立情绪参考。")
        url = os.environ.get("INDEX_TTS_API_URL", "http://127.0.0.1:18081").rstrip("/")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("INDEX_TTS_API_URL 必须为有效服务根地址。")
        request = urllib.request.Request(url + "/v1/tts", json.dumps(payload, ensure_ascii=False).encode(),
                                         {"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if parsed.hostname in ("127.0.0.1", "localhost", "::1") else urllib.request.build_opener()
        try:
            with opener.open(request, timeout=1800) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Index API HTTP {exc.code}: " + exc.read(4096).decode(errors="replace")) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Index API 连接失败：{exc}；检查 18081 服务和 index 日志。") from exc
        return (decode_audio(raw),)


NODE_CLASS_MAPPINGS = {"IndexTTSReference": Reference, "IndexTTSParameters": Parameters, "IndexTTSGenerate": Generate}
NODE_DISPLAY_NAME_MAPPINGS = {"IndexTTSReference": "Index Reference · 可留空 / 外部音频优先",
    "IndexTTSParameters": "Index 参数 · 2.0 / 2.5", "IndexTTSGenerate": "Index 克隆 / 默认音色生成"}
