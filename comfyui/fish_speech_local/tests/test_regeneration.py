import base64
import copy
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fish_speech_local.nodes import SingleVoice, MultiVoice, LoopReference
from fish_speech_local.regeneration import reference_transcript
from helpers import MockFish, response_audio, wav_bytes


def take(value):
    out = io.BytesIO()
    sf.write(out, np.full(4410, value, dtype='float32'), 22050, format='WAV', subtype='FLOAT')
    return out.getvalue()


class RegenerationTests(unittest.TestCase):
    def refs(self):
        return [{'text': '原始音频实际原文。', 'wav': wav_bytes(16000)}]

    def test_original_anchors_and_only_latest_generation_are_preserved(self):
        refs = self.refs(); before = copy.deepcopy(refs)
        parameters = {'temperature': 1., 'top_p': .9, 'single_chunk': True}
        sent = []
        def request(payload):
            sent.append(copy.deepcopy(payload))
            return take(len(sent) * .05)
        with patch('fish_speech_local.nodes.generate', side_effect=request):
            audio, = SingleVoice().run('[惊喜]你回来了！<soft>回来就好。</soft>',
                references=refs, parameters=parameters, regenerate_rounds=4)
        self.assertEqual(len(sent), 5)
        self.assertEqual([p['seed'] for p in sent], [42, 1042, 2042, 3042, 3042])
        self.assertEqual([len(p['references']) for p in sent], [1, 2, 2, 2, 2])
        for index, payload in enumerate(sent):
            self.assertEqual(payload['references'][0]['text'], refs[0]['text'])
            self.assertEqual(base64.b64decode(payload['references'][0]['audio']), refs[0]['wav'])
            self.assertEqual(payload['text'], '[惊喜]你回来了！<soft>回来就好。</soft>')
            self.assertTrue(all(r['speaker'] == 0 for r in payload['references']))
            self.assertNotIn('regenerate_rounds', payload)
            if index:
                generated = payload['references'][-1]
                self.assertEqual(generated['text'], '你回来了！回来就好。')
                data = io.BytesIO(base64.b64decode(generated['audio']))
                self.assertEqual(sf.info(data).subtype, 'PCM_24'); data.seek(0)
                samples, rate = sf.read(data)
                self.assertEqual(rate, 22050)
                np.testing.assert_allclose(samples, index * .05, atol=2e-7)
        self.assertTrue(torch.allclose(audio['waveform'], torch.full((1, 1, 4410), .25)))
        self.assertEqual(refs, before)
        self.assertEqual(parameters, {'temperature': 1., 'top_p': .9, 'single_chunk': True})

    def test_zero_rounds_and_empty_reference_make_one_request(self):
        with MockFish() as api:
            SingleVoice().run('你好。', references=self.refs(), regenerate_rounds=0)
            SingleVoice().run('普通生成。', regenerate_rounds=4)
            self.assertEqual(len(api.requests), 2)
            self.assertEqual(api.requests[-1][1]['references'], [])

    def test_stream_and_lossy_settings_only_apply_to_final_response(self):
        for options in ({'format': 'mp3'}, {'format': 'wav', 'streaming': True}, {'format': 'pcm'}):
            with self.subTest(options=options), MockFish() as api:
                SingleVoice().run('你好。', references=self.refs(), parameters=options, regenerate_rounds=2)
                sent = [p[1] for p in api.requests]
                self.assertEqual([p['format'] for p in sent[:2]], ['wav', 'wav'])
                self.assertTrue(all(not p['streaming'] for p in sent[:2]))
                self.assertEqual(sent[-1]['format'], options['format'])
                self.assertEqual(sent[-1]['streaming'], options.get('streaming', False))

    def test_invalid_settings_and_tag_only_scripts_fail_before_requests(self):
        with MockFish() as api:
            for rounds in (-1, 5, True, 1.5):
                with self.subTest(rounds=rounds), self.assertRaisesRegex(ValueError, '0–4'):
                    SingleVoice().run('你好。', references=self.refs(), regenerate_rounds=rounds)
            with self.assertRaisesRegex(ValueError, 'reference_id'):
                SingleVoice().run('你好。', parameters={'reference_id': 'voice'}, regenerate_rounds=1)
            with self.assertRaisesRegex(ValueError, '实际台词'):
                SingleVoice().run('[大笑]<soft></soft>', references=self.refs(), regenerate_rounds=1)
            with self.assertRaisesRegex(ValueError, '原生多角色'):
                MultiVoice().run('小明：你好。', references=self.refs(), regenerate_rounds=1)
            self.assertEqual(api.requests, [])

    def test_seed_source_none_and_wrap(self):
        for seed, expected in [(None, [None]*5), (2**32-1, [2**32-1, 999, 1999, 2999, 2999])]:
            with self.subTest(seed=seed), MockFish() as api:
                SingleVoice().run('你好。', references=self.refs(), seed=8,
                                  parameters={'seed': seed}, regenerate_rounds=4)
                self.assertEqual([p[1].get('seed') for p in api.requests], expected)

    def test_nine_original_clips_are_kept_and_not_replaced(self):
        references = self.refs()*9
        with MockFish() as api:
            SingleVoice().run('你好。', references=references, regenerate_rounds=1)
            self.assertEqual([len(p[1]['references']) for p in api.requests], [9, 10])
            self.assertTrue(all(x['speaker']==0 for x in api.requests[-1][1]['references']))

    def test_loop_rounds_finish_per_line_without_cross_role_feedback(self):
        refs = [{'text': '小明：男声原文。', 'wav': wav_bytes(16000)},
                {'text': '小红：女声原文。', 'wav': wav_bytes(24000)}]
        with MockFish() as api:
            full, parts = LoopReference().run('小明：[惊喜]你回来了。\n小红：回来就好。', refs,
                                              gap_ms=0, regenerate_rounds=4)
            sent = [p[1] for p in api.requests]
        self.assertEqual(len(sent), 10)
        self.assertEqual(len(parts), 2)
        self.assertEqual(full['waveform'].shape[-1], sum(p['waveform'].shape[-1] for p in parts))
        for group, original, body, seed in [(sent[:5], refs[0], '你回来了。', 42),
                                             (sent[5:], refs[1], '回来就好。', 43)]:
            self.assertEqual([p['seed'] for p in group], [seed, seed+1000, seed+2000, seed+3000, seed+3000])
            self.assertEqual([len(p['references']) for p in group], [1, 2, 2, 2, 2])
            self.assertTrue(all(base64.b64decode(p['references'][0]['audio'])==original['wav'] for p in group))
            self.assertTrue(all(p['references'][-1]['text']==body for p in group[1:]))
            self.assertTrue(all(r['speaker']==0 for p in group for r in p['references']))

    def test_every_line_transcript_validated_before_any_request(self):
        with MockFish() as api, self.assertRaisesRegex(ValueError, '第 2 行.*实际台词'):
            LoopReference().run('小明：你好。\n小明：[大笑]',
                [{'text': '小明：原文。', 'wav': wav_bytes()}], regenerate_rounds=1)
        self.assertEqual(api.requests, [])

    def test_failure_does_not_retry_or_fall_back_to_an_earlier_round(self):
        calls = []
        def generate(payload):
            calls.append(payload)
            if len(calls)==2:raise RuntimeError('test failure')
            return wav_bytes()
        with patch('fish_speech_local.nodes.generate', side_effect=generate), self.assertRaisesRegex(RuntimeError, '第 1/4 轮请求失败'):
            SingleVoice().run('你好。', references=self.refs(), regenerate_rounds=4)
        self.assertEqual(len(calls), 2)

    def test_cancel_between_rounds_is_preserved(self):
        class Cancelled(Exception): pass
        count = [0]
        def check():
            count[0] += 1
            if count[0]==3:raise Cancelled('stopped before the second request')
        management=types.ModuleType('comfy.model_management')
        management.throw_exception_if_processing_interrupted=check
        utils=types.ModuleType('comfy.utils');updates=[]
        utils.ProgressBar=lambda total: types.SimpleNamespace(update=lambda n: updates.append(n))
        modules={'comfy':types.ModuleType('comfy'),'comfy.model_management':management,'comfy.utils':utils}
        with MockFish() as api, patch.dict(sys.modules, modules), self.assertRaises(Cancelled):
            SingleVoice().run('你好。', references=self.refs(), regenerate_rounds=4)
        self.assertEqual(len(api.requests),1);self.assertEqual(updates,[1])

    def test_known_and_multiline_direction_tags_are_not_reference_words(self):
        self.assertEqual(reference_transcript('[惊喜，\n声音发颤]你回来啦！<soft>太好了。</soft>'), '你回来啦！太好了。')

    def test_native_multi_has_no_regeneration_widget(self):
        self.assertNotIn('regenerate_rounds', MultiVoice.INPUT_TYPES()['optional'])
        self.assertIn('regenerate_rounds', SingleVoice.INPUT_TYPES()['optional'])
        self.assertIn('regenerate_rounds', LoopReference.INPUT_TYPES()['optional'])


if __name__=='__main__':unittest.main(verbosity=2)
