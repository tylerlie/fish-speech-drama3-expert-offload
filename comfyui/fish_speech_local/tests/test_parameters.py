import ast
import copy
import io
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fish_speech_local.audio import decode_audio
from fish_speech_local.core import build_payload, generate
from fish_speech_local.nodes import SingleVoice, MultiVoice, LoopReference
from fish_speech_local.parameters import APIParameters, API_FIELDS, validate_parameters
from helpers import MockFish, response_audio, wav_bytes


class ParameterTests(unittest.TestCase):
    def params(self, **overrides):
        return APIParameters().configure(**overrides)[0]

    def test_defaults_match_requested_comparison(self):
        p = self.params()
        self.assertEqual((p['temperature'], p['top_p'], p['max_new_tokens']), (1., .9, 1024))
        self.assertNotIn('seed', p)
        self.assertNotIn('reference_id', p)
        self.assertEqual(p['format'], 'wav')

    def test_covers_every_self_hosted_request_field(self):
        fish = Path(os.getenv('FISH_SOURCE', str(Path(__file__).resolve().parents[2]/'fish-speech-drama3-expert-offload')))
        schema = fish/'fish_speech/utils/schema.py'
        if not schema.exists(): self.skipTest('Set up a local Fish source checkout to compare its schema.')
        node = next(n for n in ast.parse(schema.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='ServeTTSRequest')
        fields = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
        self.assertEqual(API_FIELDS, fields - {'text', 'references'})

    def test_all_sampling_and_compatibility_fields_reach_payload(self):
        p = self.params(temperature=.85, top_p=.95, max_new_tokens=600, single_chunk=True,
            use_memory_cache='on', chunk_length=450, repetition_penalty=1.5, normalize=False, latency='balanced')
        sent = build_payload('测试。', parameters=p)
        self.assertTrue(all(sent[k] == v for k, v in p.items()))
        self.assertEqual(sent['seed'], 42)

    def test_parameter_validation_rejects_invalid_or_hosted_only_fields(self):
        cases = [{'temperature': float('nan')}, {'top_p': .05}, {'temperature': True},
                 {'max_new_tokens': 1.5}, {'chunk_length': 99}, {'normalize': 'true'},
                 {'streaming': 1}, {'format': 'flac'}, {'latency': 'low'}, {'use_memory_cache': 'yes'},
                 {'seed': -1}, {'sample_rate': 44100}, {'min_chunk_length': 50},
                 {'reference_id': '../voice'}, {'reference_id': 'http://voice'}]
        for p in cases:
            with self.subTest(p=p), self.assertRaises(ValueError): validate_parameters(p)

    def test_streaming_requires_wav_before_any_http_request(self):
        with MockFish() as api:
            for fmt in ('mp3', 'opus', 'pcm'):
                with self.subTest(fmt=fmt), self.assertRaisesRegex(ValueError, '只支持 WAV'):
                    SingleVoice().run('测试。', parameters={'format': fmt, 'streaming': True})
            self.assertEqual(api.requests, [])

    def test_seed_selection_and_old_graph_compatibility(self):
        self.assertEqual(build_payload('测试。', seed=123, parameters=self.params())['seed'], 123)
        p = self.params(seed_mode='参数节点种子', seed=99)
        self.assertEqual(build_payload('测试。', seed=123, parameters=p)['seed'], 99)
        self.assertNotIn('seed', build_payload('测试。', parameters=self.params(seed_mode='不指定种子')))
        old = build_payload('测试。', seed=123)
        self.assertNotIn('temperature', old)
        self.assertNotIn('top_p', old)

    def test_nullable_sampling_fields_remain_server_defaults(self):
        p = {'temperature': None, 'top_p': None, 'max_new_tokens': None, 'repetition_penalty': None}
        self.assertTrue(all(build_payload('测试。', parameters=p)[k] is None for k in p))

    def test_saved_voice_id_only_in_unambiguous_single_mode(self):
        p = self.params(reference_id='saved_voice-1')
        self.assertEqual(build_payload('测试。', parameters=p)['reference_id'], 'saved_voice-1')
        refs = [{'text': '小明：原文。', 'wav': wav_bytes()}]
        with self.assertRaisesRegex(ValueError, 'Reference 全空'):
            build_payload('测试。', refs, parameters=p)
        with MockFish() as api:
            with self.assertRaises(ValueError): MultiVoice().run('小明：测试。', references=refs, parameters=p)
            with self.assertRaises(ValueError): LoopReference().run('小明：测试。', refs, parameters=p)
            self.assertEqual(api.requests, [])

    def test_one_configuration_reused_by_all_three_generators(self):
        p = self.params(seed_mode='参数节点种子', seed=100, temperature=.9, top_p=.95)
        original = copy.deepcopy(p)
        refs = [{'text': '小明：原文。', 'wav': wav_bytes()}]
        with MockFish() as api:
            SingleVoice().run('单音色。', seed=7, references=refs, parameters=p)
            MultiVoice().run('小明：多角色。', seed=7, references=refs, parameters=p)
            LoopReference().run('小明：第一行。\n小明：第二行。', refs, seed=7, parameters=p)
            sent = [record[1] for record in api.requests]
        self.assertEqual([x['seed'] for x in sent], [100, 100, 100, 101])
        self.assertTrue(all(x['temperature']==.9 and x['top_p']==.95 for x in sent))
        self.assertEqual(p, original)

    def test_loop_omits_seed_when_requested_and_wraps_explicit_seed(self):
        refs = [{'text': '小明：原文。', 'wav': wav_bytes()}]
        with MockFish() as api:
            LoopReference().run('小明：一句。\n小明：二句。', refs,
                parameters=self.params(seed_mode='不指定种子'))
            self.assertTrue(all('seed' not in p[1] for p in api.requests))
            LoopReference().run('小明：一句。\n小明：二句。', refs,
                parameters=self.params(seed_mode='参数节点种子', seed=2**32-1))
            self.assertEqual([p[1]['seed'] for p in api.requests[-2:]], [2**32-1, 0])

    def test_http_formats_decode_to_standard_cpu_audio(self):
        with MockFish() as api:
            for fmt in ('wav', 'mp3', 'opus', 'pcm'):
                with self.subTest(fmt=fmt):
                    audio, = SingleVoice().run('测试。', parameters=self.params(format=fmt))
                    self.assertEqual(audio['waveform'].device.type, 'cpu')
                    self.assertEqual(tuple(audio['waveform'].shape[:2]), (1, 1))
                    self.assertGreater(audio['waveform'].shape[-1], 1000)
                    self.assertEqual(api.requests[-1][1]['format'], fmt)
                    if fmt == 'pcm': self.assertEqual(audio['sample_rate'], 44100)

    def test_streaming_wav_decodes_all_samples_and_loop_joins(self):
        p = self.params(streaming=True)
        refs = [{'text': '小明：原文。', 'wav': wav_bytes()}]
        with MockFish():
            audio, = SingleVoice().run('测试。', parameters=p)
            np.testing.assert_array_equal(audio['waveform'].numpy(), decode_audio(wav_bytes())['waveform'].numpy())
            full, parts = LoopReference().run('小明：一。\n小明：二。', refs, gap_ms=0, parameters=p)
            self.assertEqual(full['waveform'].shape[-1], sum(x['waveform'].shape[-1] for x in parts))

    def test_bad_audio_in_each_format_is_not_returned_as_success(self):
        with MockFish() as api:
            api.bad_audio = True
            for fmt in ('wav', 'mp3', 'opus', 'pcm'):
                with self.subTest(fmt=fmt), self.assertRaisesRegex(RuntimeError, '有效'):
                    SingleVoice().run('测试。', parameters=self.params(format=fmt))

    def test_partial_stream_and_empty_pcm_are_errors(self):
        with self.assertRaises(RuntimeError): decode_audio(response_audio(streaming=True)[:-1], streaming=True)
        with self.assertRaises(RuntimeError): decode_audio(b'', audio_format='pcm')


if __name__ == '__main__': unittest.main(verbosity=2)
