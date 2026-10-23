"""CPU audio exchange; an entirely empty reference is valid."""
import io
import os
from pathlib import Path

MEDIA_EXTENSIONS = frozenset({'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac', '.opus',
    '.aif', '.aiff', '.wma', '.wv', '.mp4', '.webm', '.mov', '.mkv', '.avi'})


def input_directory():
    import folder_paths
    return Path(folder_paths.get_input_directory()).resolve()


def available_reference_media():
    """List reusable input media, including subfolders, without leaving input/."""
    base = input_directory()
    files = []
    for directory, subdirs, names in os.walk(base, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith('.'))
        for name in names:
            path = Path(directory) / name
            if name.startswith('.') or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(base) and resolved.is_file():
                files.append(path.relative_to(base).as_posix())
    return ['', *sorted(files, key=lambda name: (name.casefold(), name))]


def validate_reference(audio=''):
    if not isinstance(audio, str):
        raise ValueError('音频文件名必须是文本。')
    audio = audio.strip()
    if not audio: return None
    base = input_directory()
    name = audio.replace('\\', '/')
    path = (base / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(base) or not path.is_file():
        raise ValueError('参考音频不存在或位置无效，请重新上传；不能当作空参考跳过。')
    return path


def encode_reference(path):
    import numpy as np
    import soundfile as sf
    try:
        samples, rate = sf.read(path, dtype='float32', always_2d=True)
    except (RuntimeError, ValueError):
        import av
        try:
            with av.open(str(path)) as container:
                if not container.streams.audio: raise ValueError('没有音频流')
                stream = container.streams.audio[0]
                rate = stream.codec_context.sample_rate
                converter = av.AudioResampler(format='fltp', layout=stream.codec_context.layout, rate=rate)
                frames = []
                for frame in container.decode(stream):
                    frames.extend(f.to_ndarray().T for f in converter.resample(frame))
                frames.extend(f.to_ndarray().T for f in converter.resample(None))
                samples = np.concatenate(frames, axis=0) if frames else np.zeros((0, stream.channels))
        except Exception as exc:
            raise ValueError('参考音频无法解码，请使用有效 WAV、FLAC 或受支持的音频文件。') from exc
    if rate <= 0 or len(samples) < rate / 10: raise ValueError('参考音频至少需要 0.1 秒。')
    if not np.isfinite(samples).all(): raise ValueError('参考音频包含无效采样值。')
    data = io.BytesIO()
    sf.write(data, samples.mean(axis=1).clip(-1, 1), rate, format='FLAC', subtype='PCM_24')
    return data.getvalue()


def encode_audio_input(audio):
    """Encode one standard ComfyUI AUDIO on CPU, with no file fallback."""
    from numbers import Integral
    import numpy as np
    import soundfile as sf
    import torch
    if not isinstance(audio, dict):
        raise ValueError('上游未提供有效 AUDIO，请检查音频节点或断开 audio_in 后使用本地上传。')
    waveform, rate = audio.get('waveform'), audio.get('sample_rate')
    if not isinstance(rate, Integral) or isinstance(rate, bool) or rate <= 0:
        raise ValueError('上游 AUDIO 的 sample_rate 必须为正整数。')
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError('上游 AUDIO 的 waveform 必须是 [批次, 声道, 采样] 三维张量。')
    if waveform.shape[0] != 1 or waveform.shape[1] < 1:
        raise ValueError('每个 Reference 只能接收一段音频（批次为 1，至少一个声道）。')
    if not waveform.is_floating_point():
        raise ValueError('上游 AUDIO 必须提供浮点采样值。')
    samples = waveform.detach().to(device='cpu', dtype=torch.float32)[0].numpy()
    if samples.shape[1] < rate / 10:
        raise ValueError('参考音频至少需要 0.1 秒。')
    if not np.isfinite(samples).all():
        raise ValueError('参考音频包含无效采样值。')
    data = io.BytesIO()
    sf.write(data, samples.mean(axis=0).clip(-1, 1), int(rate), format='FLAC', subtype='PCM_24')
    return data.getvalue()


def decode_audio(data):
    import numpy as np
    import soundfile as sf
    import torch
    samples, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise RuntimeError('Index 返回空音频或无效采样值。')
    return {'waveform': torch.from_numpy(samples.T.copy()).unsqueeze(0), 'sample_rate': rate}
