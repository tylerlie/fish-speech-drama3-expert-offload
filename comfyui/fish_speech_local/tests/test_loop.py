import base64
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fish_speech_local.audio import decode_audio
from fish_speech_local.loop import concatenate_audio, plan_lines
from fish_speech_local.nodes import LoopReference
from helpers import MockFish, wav_bytes


class LoopTests(unittest.TestCase):
    def refs(self):
        return [{'text': '小明：[开心]参考原文一。', 'wav': wav_bytes()},
                {'text': '小红：参考原文二。', 'wav': wav_bytes(16000)},
                {'text': '小明：另一条参考。', 'wav': wav_bytes(24000)}]

    def test_requests_are_single_voice_and_ordered(self):
        refs = self.refs(); before = copy.deepcopy(refs)
        with MockFish() as api:
            full, segments = LoopReference().run(
                '小红：[开心]Bonjour.\n小明：[低声]こんにちは。\n小红：时间：12:30。', refs, seed=42, gap_ms=150)
            sent = [item[1] for item in api.requests]
        self.assertEqual([p['text'] for p in sent], ['[happy]Bonjour.', '[whisper]こんにちは。', '时间：12:30。'])
        self.assertEqual([p['seed'] for p in sent], [42, 43, 44])
        self.assertEqual([len(p['references']) for p in sent], [1, 2, 1])
        self.assertTrue(all(r['speaker'] == 0 for p in sent for r in p['references']))
        self.assertEqual(sent[1]['references'][0]['text'], '[开心]参考原文一。')
        self.assertEqual(base64.b64decode(sent[0]['references'][0]['audio']), refs[1]['wav'])
        self.assertTrue(all('<|speaker:' not in p['text'] for p in sent))
        self.assertEqual(len(segments), 3)
        self.assertEqual(refs, before)
        self.assertEqual(full['waveform'].device.type, 'cpu')

    def test_blank_lines_crlf_and_original_line_numbers(self):
        lines, roles = plan_lines('\ufeff\r\n 小明: 你好。 \r\n\r\n 小红：[开心]好。\r\n', self.refs())
        self.assertEqual([(p.number, p.role, p.text) for p in lines], [(2, '小明', '你好。'), (4, '小红', '[开心]好。')])
        self.assertEqual(list(roles), ['小明', '小红'])

    def test_all_lines_validated_before_any_request(self):
        cases = [('小明：你好。\n\n小王：好。', '第 3 行.*小王'),
                 ('小明：你好。\n继续说。', '第 2 行缺少角色名'),
                 ('小明：你好。\n小红：', '第 2 行.*没有台词'),
                 ('小明：你好。\n小红：<|speaker:0|>坏。', '内部'),
                 ('\n\t\n', '为空'), ('\ufeff', '为空')]
        with MockFish() as api:
            for text, error in cases:
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, error):
                    LoopReference().run(text, self.refs())
            self.assertEqual(api.requests, [])

    def test_empty_invalid_and_excess_references(self):
        cases = [([], '至少需要一个'), ([None], '参考无效'),
                 ([{'text': '小明：', 'wav': b'valid'}], '缺少参考原文'),
                 ([{'text': '普通原文', 'wav': b'valid'}], '缺少角色名'),
                 (self.refs()*4, '最多')]
        with MockFish() as api:
            for refs, error in cases:
                with self.subTest(refs=refs), self.assertRaisesRegex(ValueError, error):
                    LoopReference().run('小明：你好。', refs)
            self.assertEqual(api.requests, [])

    def test_nine_roles_and_unbounded_number_of_lines(self):
        refs = [{'text': f'角色{i}：原文。', 'wav': wav_bytes()} for i in range(1, 10)]
        script = '\n'.join(f'角色{i%9+1}：第{i}句。' for i in range(27))
        lines, roles = plan_lines(script, refs)
        self.assertEqual(len(lines), 27); self.assertEqual(len(roles), 9)

    def test_failure_stops_without_partial_output(self):
        with MockFish() as api:
            api.fail_texts.add('失败句。')
            with self.assertRaisesRegex(RuntimeError, '第 3 行（小红）.*HTTP 500'):
                LoopReference().run('小明：第一句。\n\n小红：失败句。\n小明：不应发送。', self.refs())
            self.assertEqual([p[1]['text'] for p in api.requests], ['第一句。', '失败句。'])

    def test_bad_audio_has_source_line_context(self):
        with MockFish() as api:
            api.bad_audio = True
            with self.assertRaisesRegex(RuntimeError, '第 1 行（小红）.*WAV'):
                LoopReference().run('小红：你好。', self.refs())
            self.assertEqual(len(api.requests), 1)

    def test_exact_join_gap_and_no_segment_mutation(self):
        a = {'sample_rate': 1000, 'waveform': torch.full((1, 1, 8), .2)}
        b = {'sample_rate': 1000, 'waveform': torch.full((1, 1, 3), -.1)}
        full = concatenate_audio([a, b], 5)
        self.assertEqual(tuple(full['waveform'].shape), (1, 1, 16))
        self.assertTrue(torch.equal(full['waveform'][..., :8], a['waveform']))
        self.assertTrue(torch.equal(full['waveform'][..., 8:13], torch.zeros(1, 1, 5)))
        self.assertTrue(torch.equal(full['waveform'][..., 13:], b['waveform']))
        self.assertEqual(tuple(a['waveform'].shape), (1, 1, 8))
        self.assertEqual(concatenate_audio([a, b], 0)['waveform'].shape[-1], 11)
        self.assertTrue(torch.equal(concatenate_audio([a], 500)['waveform'], a['waveform']))

    def test_rate_and_channels_mismatch_are_explicit(self):
        a = decode_audio(wav_bytes())
        for b in (decode_audio(wav_bytes(16000)),
                  {'sample_rate': a['sample_rate'], 'waveform': a['waveform'].repeat(1, 2, 1)}):
            with self.assertRaisesRegex(ValueError, '第 2 段.*采样率或声道'):
                concatenate_audio([a, b])

    def test_invalid_gap_before_inference(self):
        with MockFish() as api:
            for gap in (-1, 5001, 1.5):
                with self.subTest(gap=gap), self.assertRaises(ValueError):
                    LoopReference().run('小明：你好。', self.refs(), gap_ms=gap)
            self.assertEqual(api.requests, [])

    def test_seed_wrap_and_one_line(self):
        with MockFish() as api:
            full, parts = LoopReference().run('小明：一句。\n小红：二句。', self.refs(), seed=2**32-1)
            self.assertEqual([p[1]['seed'] for p in api.requests], [2**32-1, 0])
            full, parts = LoopReference().run('小明：一句。', self.refs(), gap_ms=1000)
            self.assertEqual(len(parts), 1)
            self.assertTrue(torch.equal(full['waveform'], parts[0]['waveform']))

    def test_cancel_between_lines_preserves_interrupt(self):
        class Cancelled(Exception): pass
        model_management = types.ModuleType('comfy.model_management')
        count = [0]
        def check():
            count[0] += 1
            if count[0] == 3: raise Cancelled('cancel between lines')
        model_management.throw_exception_if_processing_interrupted = check
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = lambda total: types.SimpleNamespace(update=lambda amount: None)
        modules = {'comfy': types.ModuleType('comfy'), 'comfy.model_management': model_management, 'comfy.utils': utils}
        with MockFish() as api, patch.dict(sys.modules, modules), self.assertRaises(Cancelled):
            LoopReference().run('小明：一句。\n小红：二句。', self.refs())
        self.assertEqual(len(api.requests), 1)


if __name__ == '__main__': unittest.main(verbosity=2)
