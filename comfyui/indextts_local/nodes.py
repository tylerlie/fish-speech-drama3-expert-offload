"""CPU-only ComfyUI client for the companion index_api.py adapter."""
import base64
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request

from .audio import available_reference_media, validate_reference, encode_reference, encode_audio_input, decode_audio

UNCONNECTED = object()
VOICE_ONLY = "仅参考音色"
VOICE_AND_EMOTION = "参考音色＋情绪"
REFERENCE_MODES = (VOICE_ONLY, VOICE_AND_EMOTION)
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
PARAMETERS = {
    "emotion_text": ("STRING", {"default": "", "multiline": True, "tooltip": "仅参考音色时：非空使用此描述，留空才使用八维情绪节点。参考音色＋情绪时忽略描述与滑块。正文不用于自动情绪分析。"}),
    "emo_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "文字或滑块情绪的强度；参考音色＋情绪时不使用此参数。"}),
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
        return {"required": {"reference_mode": (list(REFERENCE_MODES), {"default": VOICE_ONLY})},
                "optional": {"audio": (available_reference_media(), {"default": ""}),
                             "audio_in": ("AUDIO",)}}
    RETURN_TYPES = ("INDEX_REFERENCE",)
    RETURN_NAMES = ("reference",)
    FUNCTION = "bind"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "在此选择仅参考音色，或同时参考音色和情绪。上传、下拉选择与上游 AUDIO 均支持，连线优先。音频为空时使用服务器默认参考；不需要参考原文。"

    @classmethod
    def VALIDATE_INPUTS(cls, reference_mode=VOICE_ONLY, audio="", audio_in=UNCONNECTED, input_types=None):
        try:
            input_types = input_types or {}
            if "reference_mode" not in input_types and reference_mode not in REFERENCE_MODES:
                return "请在音频节点选择有效的参考方式。"
            for name, kind in (input_types or {}).items():
                allowed = {"audio": ("STRING", "COMBO"), "reference_mode": ("STRING", "COMBO"), "audio_in": ("AUDIO",)}
                if kind not in allowed.get(name, ()):
                    return "输入类型不匹配。"
            if "audio_in" in (input_types or {}) or audio_in is not UNCONNECTED or "audio" in (input_types or {}):
                return True
            validate_reference(audio)
            return True
        except ValueError as exc:
            return str(exc)

    @classmethod
    def IS_CHANGED(cls, reference_mode=VOICE_ONLY, audio="", audio_in=UNCONNECTED):
        if audio_in is not UNCONNECTED:
            return float("nan")
        path = validate_reference(audio)
        return reference_mode + ":" + (hashlib.sha256(path.read_bytes()).hexdigest() if path else "server-default")

    def bind(self, reference_mode=VOICE_ONLY, audio="", audio_in=UNCONNECTED):
        if reference_mode not in REFERENCE_MODES:
            raise ValueError("请在音频节点选择有效的参考方式。")
        if audio_in is not UNCONNECTED:
            raw = encode_audio_input(audio_in)
        else:
            path = validate_reference(audio)
            raw = encode_reference(path) if path else None
        # Preserve the selection even when the audio uses the server default.
        return ({"audio": raw, "reference_mode": reference_mode},)


def validate_emotion_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError("八维情绪必须包含 8 个 0–1 数值。")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("八维情绪必须包含 8 个 0–1 数值。")
    return [float(v) for v in values]


class EmotionVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {name: ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05})
                             for name in EMOTIONS}}
    RETURN_TYPES = ("INDEX_EMOTION_VECTOR",)
    RETURN_NAMES = ("emotion_vector",)
    FUNCTION = "build"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "八维情绪滑块，连接参数节点；仅参考音色且 emotion_text 为空时生效。全 0 表示不额外加强任何向量情绪；服务端按官方方法归一化。"

    def build(self, **values):
        if set(values) - set(EMOTIONS):
            raise ValueError("存在未知的八维情绪参数。")
        return (validate_emotion_vector([values.get(name, 0.0) for name in EMOTIONS]),)


class Parameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": PARAMETERS, "optional": {"emotion_vector": ("INDEX_EMOTION_VECTOR",)}}
    RETURN_TYPES = ("INDEX_PARAMETERS",)
    FUNCTION = "build"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "情绪优先级：参考音色＋情绪 > 非空 emotion_text > 八维滑块。描述为空且未连接滑块时使用全 0 向量，不从正文分析情绪。"

    def build(self, emotion_vector=None, **values):
        if set(values) - PARAMETERS.keys():
            raise ValueError("Index 参数已更新，请导入新版工作流。")
        result = {key: options[1]["default"] for key, options in PARAMETERS.items()}
        result.update(values)
        # Validate the vector only when it wins the per-generator priority.
        # A shared parameters node can feed generators with different sources.
        if emotion_vector is not None:
            result["emotion_vector"] = emotion_vector
        return (result,)


def build_payload(text, seed=42, reference=None, parameters=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("生成文稿不能为空。")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("请连接 Index 参数节点。")
    if set(parameters or {}) - (set(PARAMETERS) | {"emotion_vector"}):
        raise ValueError("Index 参数已更新，请导入新版工作流。")
    payload = {**Parameters().build()[0], **(parameters or {}), "text": text, "seed": seed}
    vector = payload.pop("emotion_vector", None)
    mode = VOICE_ONLY
    if reference is not None:
        if not isinstance(reference, dict) or reference.get("reference_mode") not in REFERENCE_MODES:
            raise ValueError("请连接新版 Index 音频参考节点。")
        mode = reference["reference_mode"]
        raw = reference.get("audio")
        if raw is not None:
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("参考音频无效，请重新选择或上传。")
            payload["reference_audio"] = base64.b64encode(raw).decode("ascii")
    if mode == VOICE_AND_EMOTION:
        # Use exactly the same uploaded bytes for speaker and emotion. When no
        # bytes are supplied, native reference mode reuses the server's default
        # speaker audio for emotion as well. Parameter descriptions never apply.
        payload.update(emotion_mode="reference", emotion_text="", emo_alpha=1.0)
        if "reference_audio" in payload:
            payload.update(emotion_mode="audio", emotion_audio=payload["reference_audio"])
    else:
        description = payload["emotion_text"]
        if not isinstance(description, str):
            raise ValueError("情绪描述必须是文本。")
        if len(description) > 2000:
            raise ValueError("情绪描述不得超过 2000 字符。")
        if description.strip():
            payload.update(emotion_mode="text", emotion_text=description.strip())
        else:
            # Explicit vector mode prevents blank emotion text from triggering
            # upstream automatic analysis of the spoken body.
            payload.update(emotion_mode="vector", emotion_text="",
                           emotion_vector=validate_emotion_vector(vector if vector is not None else [0.0] * 8))
    return payload


class Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True}),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "control_after_generate": True})},
                "optional": {"reference": ("INDEX_REFERENCE",), "parameters": ("INDEX_PARAMETERS",)}}
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "IndexTTS Local"
    DESCRIPTION = "调用独立 Index API，兼容 2.0/2.5。优先跟随音频节点的音色＋情绪，否则先用非空情绪描述，描述为空才用八维滑块。正文只用于朗读；音频可空，使用服务器默认参考。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, text, seed=42, reference=None, parameters=None):
        payload = build_payload(text, seed, reference, parameters)
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


NODE_CLASS_MAPPINGS = {"IndexTTSReference": Reference, "IndexTTSEmotionVector": EmotionVector,
                       "IndexTTSParameters": Parameters, "IndexTTSGenerate": Generate}
NODE_DISPLAY_NAME_MAPPINGS = {"IndexTTSReference": "Index 音频参考 · 音色 / 音色＋情绪",
    "IndexTTSEmotionVector": "Index 八维情绪 · 描述为空时生效",
    "IndexTTSParameters": "Index 参数 · 2.0 / 2.5", "IndexTTSGenerate": "Index 克隆 / 默认音色生成"}
