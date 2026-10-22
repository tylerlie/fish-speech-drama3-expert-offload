"""Keep the original voice anchors while replacing one generated reference."""
import base64
import re

from .audio import encode_audio_input
from .core import LOGGER

ROUND_INPUT = ('INT', {'default': 0, 'min': 0, 'max': 4,
    'tooltip': '重生轮次：0 为只生成一次；4 为首次生成后再生成四次。每轮保留原始音色，并加入上一轮音频；耗时约随请求数增加。参考为空时仍只做一次普通生成。'})


def validate_rounds(rounds):
    if type(rounds) is not int or not 0 <= rounds <= 4:
        raise ValueError('重生轮次必须是 0–4 的整数。')
    return rounds


def reference_transcript(text):
    # Direction tags are not the words spoken in the generated reference.
    words = re.sub(r'\[[^\[\]]*\]', '', text)
    words = re.sub(r'</?[A-Za-z][A-Za-z0-9_-]*(?:\s[^<>]*)?>', '', words).strip()
    if not words:
        raise ValueError('双参考重生需要实际台词，不能只填写语气标签。')
    return words


def generation_rounds(payload, rounds):
    validate_rounds(rounds)
    if rounds and payload.get('reference_id'):
        raise ValueError('双参考重生需要原始音频，请清空 reference_id，改用 Reference 节点。')
    if rounds and payload['references']:
        if any(ref['speaker'] != 0 for ref in payload['references']):
            raise ValueError('双参考重生只支持单人请求；多角色请使用逐行克隆拼接节点。')
        reference_transcript(payload['text'])
        return rounds
    return 0


def run_generation(payload, rounds, *, generate_audio, decode_audio, check, progress=None):
    rounds = generation_rounds(payload, rounds)
    originals = [dict(ref) for ref in payload['references']]
    transcript = reference_transcript(payload['text']) if rounds else None
    previous = None
    for index in range(rounds + 1):
        check()
        request = dict(payload)
        request['references'] = [dict(ref) for ref in originals]
        if index:
            request['references'].append(previous)
        if rounds:
            # Match the validated 0/1/2/3/4-round experiment: the fourth
            # regeneration keeps round three's seed rather than changing it.
            if 'seed' in payload:
                request['seed'] = (payload['seed'] + min(index, 3) * 1000) % 2**32
            # Only the last response uses the chosen output codec. Intermediate
            # takes stay lossless and are not streamed through the next prompt.
            if index < rounds:
                request.update(format='wav', streaming=False)
            LOGGER.info('Fish 双参考重生：第 %d/%d 次请求，%d 段参考，种子 %s。',
                        index + 1, rounds + 1, len(request['references']), request.get('seed', '未指定'))
        try:
            data = generate_audio(request)
        except (RuntimeError, ValueError) as exc:
            if not rounds:
                raise
            raise RuntimeError(f'双参考重生第 {index}/{rounds} 轮请求失败：{exc}') from exc
        check()
        try:
            audio = decode_audio(data, request['format'], request['streaming'])
            if index < rounds:
                previous = {'audio': base64.b64encode(encode_audio_input(audio)).decode('ascii'),
                            'text': transcript, 'speaker': 0}
        except (RuntimeError, ValueError) as exc:
            if not rounds:
                raise
            raise RuntimeError(f'双参考重生第 {index}/{rounds} 轮音频无效：{exc}') from exc
        if progress is not None:
            progress.update(1)
    return audio
