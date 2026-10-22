from .audio import validate_reference, encode_reference, encode_audio_input, decode_audio, available_reference_media
from .core import clean_text, build_payload, generate
from .parameters import APIParameters, PARAMETERS, validate_parameters
from .regeneration import ROUND_INPUT, generation_rounds, reference_transcript, run_generation, validate_rounds

REFERENCE = 'DRAMA_REFERENCE'
REFERENCES = 'DRAMA_REFERENCES'
_UNCONNECTED = object()


class Reference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'optional': {
            'audio': (available_reference_media(), {'default': '', 'tooltip': '下拉选择 ComfyUI input 及子目录中已有的音频/媒体，无需重复上传。空白项或清空按钮表示留空；连接 audio_in 后忽略本地选择。'}),
            'text': ('STRING', {'default': '', 'multiline': True,
                    'tooltip': '填写实际使用音频的原文。单音色写录音实际原文；多角色写“角色名：实际原文”。'}),
            'audio_in': ('AUDIO', {'tooltip': '可接 Load Audio by URL 或其他 AUDIO 输出；连接后优先使用上游，忽略本节点上传的音频。'})}}
    RETURN_TYPES = (REFERENCE,)
    RETURN_NAMES = ('reference',)
    CATEGORY = 'Fish Speech'
    FUNCTION = 'bind'
    DESCRIPTION = '下拉复用已有媒体、上传新音频或连接 audio_in，并填写对应原文。连线优先，未连线使用本地选择；未连线且音频和原文都空即跳过。'

    @classmethod
    def VALIDATE_INPUTS(cls, audio='', text='', audio_in=_UNCONNECTED, input_types=None):
        try:
            # Upstream values do not exist during ComfyUI prompt validation.
            linked = input_types or {}
            expected = {'audio': 'STRING', 'text': 'STRING', 'audio_in': 'AUDIO'}
            for name, kind in linked.items():
                if kind != expected.get(name):
                    raise ValueError(f'{name} 需要 {expected.get(name)} 输入，收到 {kind}。')
            if 'text' in linked:
                return True  # Validate the resolved transcript in bind().
            if 'audio_in' in linked or audio_in is not _UNCONNECTED:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('缺少参考原文：请填写上游音频实际说的话。')
                clean_text(text)
                return True
            if 'audio' in linked:
                return True
            validate_reference(audio, text)
            if text.strip(): clean_text(text)
            return True
        except ValueError as exc: return str(exc)

    @classmethod
    def IS_CHANGED(cls, audio='', text='', audio_in=_UNCONNECTED):
        # Linked inputs are placeholders here; never inspect the unused local file.
        if audio_in is not _UNCONNECTED:
            return float('nan')
        import hashlib
        path = validate_reference(audio, text)
        return 'empty' if path is None else hashlib.sha256(path.read_bytes()).hexdigest()

    def bind(self, audio='', text='', audio_in=_UNCONNECTED):
        if audio_in is not _UNCONNECTED:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('缺少参考原文：请填写上游音频实际说的话。')
            return ({'text': clean_text(text), 'wav': encode_audio_input(audio_in)},)
        path = validate_reference(audio, text)
        if path is None: return (None,)
        return ({'text': clean_text(text), 'wav': encode_reference(path)},)


class References:
    @classmethod
    def INPUT_TYPES(cls):
        return {'optional': {f'reference_{i}': (REFERENCE,) for i in range(1, 10)}}
    RETURN_TYPES = (REFERENCES,)
    RETURN_NAMES = ('references',)
    CATEGORY = 'Fish Speech'
    FUNCTION = 'collect'
    DESCRIPTION = '九路并联，空 Reference 自动忽略，允许全部为空。按端口顺序收集，不要求连续填写。'

    def collect(self, **inputs):
        if set(inputs) - {f'reference_{i}' for i in range(1, 10)}: raise ValueError('最多连接九路 Reference。')
        result = []
        for i in range(1, 10):
            record = inputs.get(f'reference_{i}')
            if record is None: continue
            if not isinstance(record, dict) or not isinstance(record.get('wav'), bytes) or not record['wav']:
                raise ValueError(f'第 {i} 路 Reference 无效。')
            result.append({'text': clean_text(record.get('text')), 'wav': record['wav']})
        return (result,)


class SingleVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True}),
                             'seed': ('INT', {'default': 42, 'min': 0, 'max': 2**32 - 1, 'control_after_generate': True})},
                'optional': {'references': (REFERENCES,), 'parameters': (PARAMETERS,),
                             'regenerate_rounds': ROUND_INPUT}}
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'run'
    CATEGORY = 'Fish Speech'
    DESCRIPTION = ('参考为空即一次普通生成；有参考即克隆，可选 0–4 次双参考重生。'
                   '每次保留原始音色并加入上一轮音频，输出指定轮次的最终结果。')
    DIALOGUE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float('nan')

    def run(self, text, seed=42, references=None, parameters=None, regenerate_rounds=0):
        validate_rounds(regenerate_rounds)
        if self.DIALOGUE and regenerate_rounds:
            raise ValueError('原生多角色不执行双参考重生，请使用逐行克隆拼接工作流。')
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
            from comfy.utils import ProgressBar
        except ImportError:
            def throw_exception_if_processing_interrupted(): pass
            ProgressBar = None
        payload = build_payload(text, references, dialogue=self.DIALOGUE, seed=seed, parameters=parameters)
        rounds = generation_rounds(payload, regenerate_rounds)
        progress = ProgressBar(rounds + 1) if ProgressBar is not None else None
        return (run_generation(payload, rounds, generate_audio=generate, decode_audio=decode_audio,
                check=throw_exception_if_processing_interrupted, progress=progress),)


class MultiVoice(SingleVoice):
    DESCRIPTION = '至少一个克隆角色、最多九个。参考及文稿用“角色名：正文”，空参考自动忽略。'
    DIALOGUE = True

    @classmethod
    def INPUT_TYPES(cls):
        fields = super().INPUT_TYPES()
        fields['optional'].pop('regenerate_rounds')
        return fields


class LoopReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'text': ('STRING', {'forceInput': True}),
            'references': (REFERENCES,),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2**32 - 1, 'control_after_generate': True}),
            'gap_ms': ('INT', {'default': 200, 'min': 0, 'max': 5000, 'step': 10,
                               'tooltip': '只在完整拼接音频的相邻片段之间添加停顿，单位毫秒；分段数组保持原样。'})},
                'optional': {'parameters': (PARAMETERS,), 'regenerate_rounds': ROUND_INPUT}}
    RETURN_TYPES = ('AUDIO', 'AUDIO')
    RETURN_NAMES = ('完整拼接', '分段数组')
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = 'run'
    CATEGORY = 'Fish Speech'
    DESCRIPTION = ('按换行循环单人克隆，再依次拼接。每个非空行写“角色名：[语气]台词”；'
                   '空行跳过。可为每行独立执行 0–4 次双参考重生；始终保留该角色的原始参考。'
                   '两路输出为各行最终结果的完整拼接与 AUDIO 列表。')

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float('nan')

    def run(self, text, references, seed=42, gap_ms=200, parameters=None, regenerate_rounds=0):
        from .loop import plan_lines, concatenate_audio
        from .core import LOGGER
        if not isinstance(gap_ms, int) or not 0 <= gap_ms <= 5000:
            raise ValueError('段间停顿必须是 0–5000 毫秒的整数。')
        validate_rounds(regenerate_rounds)
        options = validate_parameters(parameters)
        if options.get('reference_id'):
            raise ValueError('循环多角色不支持共用 reference_id，请在 Reference 节点绑定每个角色。')
        base_seed = options.get('seed', seed)
        lines, roles = plan_lines(text, references)
        if regenerate_rounds:
            # Check every line before submitting the first inference request.
            for line in lines:
                try:
                    reference_transcript(line.text)
                except ValueError as exc:
                    raise ValueError(f'文稿第 {line.number} 行（{line.role}）：{exc}') from exc
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
            from comfy.utils import ProgressBar
        except ImportError:
            def throw_exception_if_processing_interrupted(): pass
            progress = None
        else:
            progress = ProgressBar(len(lines) * (regenerate_rounds + 1))
        segments = []
        for index, line in enumerate(lines):
            throw_exception_if_processing_interrupted()
            # Every request contains this role's clips only, all bound to speaker 0.
            line_options = dict(options)
            if 'seed' in options:
                line_options['seed'] = None if base_seed is None else (int(base_seed) + index) % 2**32
            payload = build_payload(line.text, roles[line.role], seed=(int(seed) + index) % 2**32,
                                    parameters=line_options)
            LOGGER.info('Fish 循环 Reference：第 %d/%d 段，文稿第 %d 行，角色 %s。',
                        index + 1, len(lines), line.number, line.role)
            try:
                audio = run_generation(payload, regenerate_rounds, generate_audio=generate,
                    decode_audio=decode_audio, check=throw_exception_if_processing_interrupted,
                    progress=progress)
            except RuntimeError as exc:
                raise RuntimeError(f'文稿第 {line.number} 行（{line.role}）生成失败：{exc}') from exc
            throw_exception_if_processing_interrupted()
            segments.append(audio)
        return concatenate_audio(segments, gap_ms), segments


NODE_CLASS_MAPPINGS = {'DramaReference': Reference, 'DramaReferences': References,
                       'DramaAPIParameters': APIParameters,
                       'DramaSingleVoice': SingleVoice, 'DramaMultiVoice': MultiVoice,
                       'DramaLoopReference': LoopReference}
NODE_DISPLAY_NAME_MAPPINGS = {'DramaReference': 'Fish · Reference 音频与原文',
    'DramaAPIParameters': 'Fish · API 参数（自部署）',
    'DramaReferences': 'Fish · References 九路汇总', 'DramaSingleVoice': 'Fish · 单音色克隆 / 生成',
    'DramaMultiVoice': 'Fish · 多角色长文本',
    'DramaLoopReference': 'Fish · 循环 Reference · 逐行克隆拼接'}
