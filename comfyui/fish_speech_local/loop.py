"""Line-based single-voice requests and CPU audio assembly."""
from dataclasses import dataclass

from .core import ROLE, clean_text, role_reference


@dataclass(frozen=True)
class Line:
    number: int
    role: str
    text: str


def plan_lines(text, references):
    """Validate the entire script before any paid/expensive inference request."""
    records = list(references or ())
    if not records:
        raise ValueError('循环 Reference 至少需要一个克隆音色，请填写参考音频和“角色名：参考原文”。')
    if len(records) > 9:
        raise ValueError('最多接入九段参考，最多形成九个角色。')
    roles = {}
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get('wav'), bytes) or not record['wav']:
            raise ValueError(f'第 {position} 段参考无效。')
        try:
            role, transcript = role_reference(record.get('text'))
        except ValueError as exc:
            raise ValueError(f'第 {position} 段参考：{exc}') from exc
        roles.setdefault(role, []).append({'text': transcript, 'wav': record['wav']})

    clean_text(text)
    normalized = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    lines = []
    for number, raw in enumerate(normalized.split('\n'), 1):
        if not raw.strip():
            continue
        match = ROLE.match(raw.strip())
        if not match:
            raise ValueError(f'文稿第 {number} 行缺少角色名；每个非空行都要写“角色名：[语气]台词”。')
        role, body = match.groups()
        role, body = role.strip(), body.strip()
        if role not in roles:
            raise ValueError(f'文稿第 {number} 行的“{role}”未绑定参考，请使用已填写参考中的角色名。')
        if not body:
            raise ValueError(f'文稿第 {number} 行（{role}）没有台词，请填写或删除该行。')
        lines.append(Line(number, role, body))
    if not lines:
        raise ValueError('生成文稿为空，请至少填写一行台词。')
    return lines, roles


def concatenate_audio(segments, gap_ms=200):
    """Keep each segment unchanged; only the joined output contains inserted gaps."""
    import torch

    if not segments:
        raise ValueError('没有可拼接的音频。')
    if not isinstance(gap_ms, int) or not 0 <= gap_ms <= 5000:
        raise ValueError('段间停顿必须是 0–5000 毫秒的整数。')
    rate = segments[0]['sample_rate']
    shape = segments[0]['waveform'].shape[:2]
    pieces = []
    gap_samples = round(rate * gap_ms / 1000)
    for index, audio in enumerate(segments):
        waveform = audio['waveform']
        if audio['sample_rate'] != rate or waveform.shape[:2] != shape:
            raise ValueError(f'第 {index + 1} 段的采样率或声道数与第一段不同，请检查 Fish API 输出。')
        if index and gap_samples:
            pieces.append(waveform.new_zeros((*shape, gap_samples)))
        pieces.append(waveform)
    return {'sample_rate': rate, 'waveform': torch.cat(pieces, dim=-1)}
