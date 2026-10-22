"""Text grammar and official local Fish API transport; no model or CUDA imports."""
from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import re
import threading
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext
from .parameters import validate_parameters

ROLE = re.compile(r'^([^\s:：\[\]<>][^:：\[\]<>]*)[:：](.*)$')
MOODS = {'开心': 'happy', '兴奋': 'excited', '生气': 'angry', '低声': 'whisper',
         '耳语': 'whisper', '大笑': 'laughing', '悲伤': 'sad'}
_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('文本为空，请填写内容。')
    if '<|' in text or '|>' in text:
        raise ValueError('请使用“角色名：正文”，不要手动填写模型内部的 <|...|> 标记。')
    return text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n').strip()


def role_reference(text: str) -> tuple[str, str]:
    text = clean_text(text)
    first, _, rest = text.partition('\n')
    match = ROLE.match(first.strip())
    if not match:
        raise ValueError('参考文本缺少角色名。示例：小明：这段音频里实际说的话。')
    name, transcript = match.groups()
    name = name.strip()
    transcript = (transcript + ('\n' + rest if rest else '')).strip()
    if not transcript:
        raise ValueError(f'角色“{name}”缺少参考原文，请填写音频里实际说的话。')
    return name, transcript


def translate_moods(text: str) -> str:
    # Only a small explicit vocabulary is translated; reference transcripts never
    # pass through this function. Unknown bracket content is left untouched.
    return re.sub(r'\[([^\]\n]+)\]', lambda m: '[' + MOODS.get(m[1], m[1]) + ']', text)


def dialogue_text(text: str, names: list[str]) -> str:
    # A script pasted with a common indentation should still work. Individual
    # indented body lines remain literal, resolving the "时间：..." ambiguity.
    if not isinstance(text, str):
        raise ValueError('生成文稿必须是文本。')
    text = textwrap.dedent(text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n'))
    clean_text(text)  # Validate without discarding leading blank lines/line numbers.
    if not names:
        raise ValueError('请先提供至少一个角色的参考音频和参考文本。')
    mapping = {name: i for i, name in enumerate(names)}
    turns: list[tuple[str, list[str]]] = []
    for line_no, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            if turns:
                turns[-1][1].append('')
            continue
        # Indentation explicitly makes a literal continuation, including "时间：...".
        match = None if line[:1].isspace() else ROLE.match(line)
        if match:
            name, body = match.groups()
            name = name.strip()
            if name in mapping:
                turns.append((name, [body.strip()]))
                continue
            # These colon forms are body text, not probable character declarations.
            literal = name.isdigit() or name.lower() in {'http', 'https', 'ftp'}
            if not literal:
                raise ValueError(f'生成文稿第 {line_no} 行的“{name}”未绑定参考。请增加对应参考组；'
                                 '如果这里是正文冒号，请在该行开头加一个空格。')
        if not turns:
            raise ValueError(f'第 {line_no} 行请先指定角色，例如：{names[0]}：你好。')
        turns[-1][1].append(line.lstrip())
    result = []
    for name, lines in turns:
        body = '\n'.join(lines).strip()
        if not body:
            raise ValueError(f'角色“{name}”有一段空白发言，请填写正文或删除该段。')
        result.append(f'<|speaker:{mapping[name]}|>' + translate_moods(body))
    if not result:
        raise ValueError('生成文稿为空。')
    return '\n'.join(result)


def build_payload(text, references=None, dialogue=False, seed=42, parameters=None):
    options = validate_parameters(parameters)
    records = list(references or ())
    if options.get('reference_id') and (dialogue or records):
        raise ValueError('reference_id 仅用于单音色且 Reference 全空；多角色请使用角色参考节点，避免覆盖角色音色。')
    if len(records) > 9:
        raise ValueError('最多接入 9 段参考，最多形成 9 个角色。')
    if dialogue and not records:
        raise ValueError('多角色生成至少需要一个克隆音色，请上传音频并填写“角色名：实际原文”。')
    prepared, names = [], []
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get('wav'), bytes) or not record['wav']:
            raise ValueError(f'第 {position} 段参考无效。')
        try:
            name, transcript = role_reference(record['text']) if dialogue else ('', clean_text(record['text']))
        except ValueError as exc:
            raise ValueError(f'第 {position} 段参考：{exc}') from exc
        if name not in names: names.append(name)
        prepared.append({'audio': base64.b64encode(record['wav']).decode('ascii'),
                         'text': transcript, 'speaker': names.index(name)})
    generated = dialogue_text(text, names) if dialogue else translate_moods(clean_text(text))
    payload = {'text': generated, 'references': prepared, 'format': 'wav',
            'streaming': False, 'single_chunk': False, 'seed': int(seed),
            'use_memory_cache': 'off', 'normalize': True}
    payload.update(options)
    if payload.get('seed') is None:
        payload.pop('seed')
    return payload


def request_api(path: str, payload=None, timeout=30, base_url=None, api_key=None):
    base_url = (base_url or os.getenv('FISH_API_URL', 'http://127.0.0.1:18080')).rstrip('/')
    api_key = os.getenv('FISH_API_KEY', '') if api_key is None else api_key
    address = urllib.parse.urlsplit(base_url)
    if address.scheme not in ('http', 'https') or not address.hostname or address.query or address.fragment:
        raise ValueError('FISH_API_URL 应为服务根地址，例如 http://127.0.0.1:18080。')
    if address.path.rstrip('/'):
        raise ValueError('FISH_API_URL 请只填写服务根地址，不要附加 /v1/tts。')
    try:
        address.port
    except ValueError as exc:
        raise ValueError('FISH_API_URL 的端口无效。') from exc
    content_types = {'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'opus': 'audio/ogg', 'pcm': 'application/octet-stream'}
    headers = {'Accept': content_types.get(payload.get('format', 'wav'), 'audio/wav') if payload is not None else 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(base_url + path, data=data, headers=headers)
    # Local traffic must not inherit an external cloud HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (_LOCK if payload is not None else nullcontext()):
            with opener.open(req, timeout=timeout) as response:
                return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode('utf-8', errors='replace')
        raise RuntimeError(f'Fish API 返回 HTTP {exc.code}：{detail}。请查看 fish.log。') from exc
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise RuntimeError('无法完成 Fish API 请求，请检查服务状态或等待时间。'
                           '超时或取消不会保证终止后端推理；确认完成后再重试。') from exc


def generate(payload):
    timeout = float(os.getenv('FISH_REQUEST_TIMEOUT', '7200'))
    if timeout <= 0:
        raise ValueError('FISH_REQUEST_TIMEOUT 必须是大于 0 的秒数。')
    LOGGER.info('Fish 本机 API：提交 %d 字符、%d 段参考。', len(payload['text']), len(payload['references']))
    LOGGER.info('Fish API 请求参数：%s', json.dumps(
        {k: v for k, v in payload.items() if k not in ('text', 'references', 'reference_id')}, ensure_ascii=False, sort_keys=True))
    start = time.perf_counter()
    audio = request_api('/v1/tts', payload, timeout=timeout)
    fmt = payload.get('format', 'wav')
    valid = {'wav': len(audio) > 44 and audio[:4] in (b'RIFF', b'RF64') and audio[8:12] == b'WAVE',
             'mp3': len(audio) > 4 and (audio.startswith(b'ID3') or audio[0] == 255 and audio[1] & 224 == 224),
             'opus': len(audio) > 4 and audio.startswith(b'OggS'),
             'pcm': len(audio) >= 2 and len(audio) % 2 == 0}.get(fmt, False)
    if fmt == 'pcm' and audio.lstrip().startswith((b'{', b'[')):
        try: json.loads(audio)
        except (ValueError, UnicodeError): pass
        else: valid = False
    if not valid:
        raise RuntimeError(f'Fish 没有返回有效 {fmt.upper()} 音频，请查看 fish.log。')
    LOGGER.info('Fish 本机 API：收到 %s，耗时 %.2f 秒。', fmt.upper(), time.perf_counter() - start)
    return audio
