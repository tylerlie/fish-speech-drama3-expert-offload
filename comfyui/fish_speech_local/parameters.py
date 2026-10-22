"""Request configuration for the self-hosted Fish /v1/tts API only."""
import math
import re

PARAMETERS = 'DRAMA_API_PARAMETERS'
# text and references belong to the existing workflow inputs.
API_FIELDS = frozenset({
    'temperature', 'top_p', 'max_new_tokens', 'repetition_penalty',
    'chunk_length', 'single_chunk', 'format', 'streaming', 'latency',
    'normalize', 'use_memory_cache', 'reference_id', 'seed',
})
PARAMETER_INPUTS = {
    'temperature': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 1.0, 'step': 0.05,
        'tooltip': '采样温度。接口范围 0.1–1.0；本节点默认 1.0，服务原始默认 0.7。'}),
    'top_p': ('FLOAT', {'default': 0.9, 'min': 0.1, 'max': 1.0, 'step': 0.05,
        'tooltip': '核采样概率。接口范围 0.1–1.0；本节点默认 0.9，服务原始默认 0.7。'}),
    'max_new_tokens': ('INT', {'default': 1024, 'min': 0, 'max': 2147483647,
        'tooltip': '每个内部文本块的生成帧数上限，仍受模型上下文限制。0 使用模型内置上限，不代表无限。'}),
    'seed_mode': (['生成节点种子', '参数节点种子', '不指定种子'],
        {'tooltip': '默认沿用生成节点种子。参数节点种子可让三种生成节点共用同一值；循环逐段加 1。'}),
    'seed': ('INT', {'default': 42, 'min': 0, 'max': 2**32 - 1, 'control_after_generate': False,
        'tooltip': '仅 seed_mode 为“参数节点种子”时生效；否则此数值不发送。'}),
    'single_chunk': ('BOOLEAN', {'default': False,
        'tooltip': 'Drama 3：整次请求一次生成，关闭内部文本分块。长文仍受上下文容量限制；循环模式作用于每一行。'}),
    'use_memory_cache': (['off', 'on'], {'default': 'off',
        'tooltip': '缓存参考音频编码，不是生成结果缓存，也不是跨请求的前文音频上下文。'}),
    'reference_id': ('STRING', {'default': '',
        'tooltip': '可选：自部署 Fish 中已保存的音色 ID。仅供单音色使用，填入后须清空 Reference；不能填写官网音色 ID。'}),
    'format': (['wav', 'mp3', 'opus', 'pcm'], {'default': 'wav',
        'tooltip': 'Fish HTTP 响应编码；插件解码后统一输出 AUDIO，实际保存格式在保存节点选择。PCM 按本项目 Drama 3 的 44100Hz 单声道16位解码。'}),
    'streaming': ('BOOLEAN', {'default': False,
        'tooltip': '仅 WAV 支持。接收完成后才输出 ComfyUI AUDIO，不代表边生成边试听。'}),
    'chunk_length': ('INT', {'default': 300, 'min': 100, 'max': 1000,
        'tooltip': '接口接受；当前 Drama 3 忽略此值，使用内部约300 UTF-8字节分块。S2-Pro 才使用该参数。'}),
    'repetition_penalty': ('FLOAT', {'default': 1.2, 'min': 0.9, 'max': 2.0, 'step': 0.05,
        'tooltip': '兼容字段：当前 Drama 3/S2-Pro 忽略此值，使用内部重复感知采样。'}),
    'normalize': ('BOOLEAN', {'default': True,
        'tooltip': '接口接受；当前开源推理路径未执行数字、日期规范化，也不是响度归一化。'}),
    'latency': (['normal', 'balanced'], {'default': 'normal',
        'tooltip': '接口接受；当前开源推理路径未使用，不能据此切换质量/速度模式。'}),
}


def validate_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError('API 参数必须来自 Fish API 参数节点。')
    unknown = set(parameters) - API_FIELDS
    if unknown:
        raise ValueError('自部署 Fish 不支持这些参数：' + ', '.join(sorted(map(str, unknown))))
    result = dict(parameters)
    for name, low, high in [('temperature', .1, 1.), ('top_p', .1, 1.), ('repetition_penalty', .9, 2.)]:
        if name not in result or result[name] is None:
            continue  # Null sampling fields mean the server's own defaults.
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 必须为 {low}–{high} 范围内的有限数值。')
        result[name] = float(value)
    for name, low, high in [('max_new_tokens', -2147483648, 2147483647),
                            ('chunk_length', 100, 1000), ('seed', 0, 2**32 - 1)]:
        if name not in result or result[name] is None and name in ('max_new_tokens', 'seed'):
            continue
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(f'{name} 必须为 {low}–{high} 范围内的整数。')
    for name in ('single_chunk', 'streaming', 'normalize'):
        if name in result and type(result[name]) is not bool:
            raise ValueError(f'{name} 必须是布尔值。')
    for name, choices in [('format', ('wav', 'pcm', 'mp3', 'opus')),
                          ('latency', ('normal', 'balanced')), ('use_memory_cache', ('on', 'off'))]:
        if name in result and result[name] not in choices:
            raise ValueError(f'{name} 必须为 ' + ' / '.join(choices) + '。')
    if result.get('streaming') and result.get('format', 'wav') != 'wav':
        raise ValueError('自部署 Fish 的 streaming 只支持 WAV；请关闭流式或选择 wav。')
    if 'reference_id' in result:
        value = result['reference_id']
        if value is not None and not isinstance(value, str):
            raise ValueError('reference_id 必须是文本。')
        value = (value or '').strip()
        if value and (len(value) > 255 or re.fullmatch(r'[A-Za-z0-9_-]+', value) is None):
            raise ValueError('reference_id 仅支持字母、数字、下划线、连字符，最多255字符。')
        if value:
            result['reference_id'] = value
        else:
            result.pop('reference_id')
    return result


class APIParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': PARAMETER_INPUTS}

    RETURN_TYPES = (PARAMETERS,)
    RETURN_NAMES = ('parameters',)
    FUNCTION = 'configure'
    CATEGORY = 'Fish Speech'
    DESCRIPTION = ('自部署 /v1/tts 参数，可共用并连接三个生成节点。正文和参考由原节点输入。'
        '连接后参数优先，种子默认沿用生成节点。chunk_length、重复惩罚、normalize、latency '
        '为接口兼容字段，当前 Drama 3 未实现其调节效果。')

    def configure(self, temperature=1.0, top_p=.9, max_new_tokens=1024,
                  seed_mode='生成节点种子', seed=42, single_chunk=False,
                  use_memory_cache='off', reference_id='', format='wav', streaming=False,
                  chunk_length=300, repetition_penalty=1.2, normalize=True, latency='normal'):
        if seed_mode not in ('生成节点种子', '参数节点种子', '不指定种子'):
            raise ValueError('未知种子模式。')
        params = dict(temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
            single_chunk=single_chunk, use_memory_cache=use_memory_cache, reference_id=reference_id,
            format=format, streaming=streaming, chunk_length=chunk_length,
            repetition_penalty=repetition_penalty, normalize=normalize, latency=latency)
        if seed_mode != '生成节点种子':
            params['seed'] = seed if seed_mode == '参数节点种子' else None
        return (validate_parameters(params),)
