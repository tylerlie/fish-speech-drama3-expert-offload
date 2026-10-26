import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from indextts_local.nodes import EmotionVector, Generate, Parameters, Reference, VOICE_AND_EMOTION, VOICE_ONLY


def wav_bytes(seconds=.2):
    data = io.BytesIO()
    sf.write(data, np.sin(np.arange(round(24000 * seconds)) * .03) * .1, 24000, format='WAV')
    return data.getvalue()


class ReferenceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        raw = wav_bytes()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                cls.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args): pass
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        env = patch.dict(os.environ, INDEX_TTS_API_URL=f'http://127.0.0.1:{self.server.server_port}')
        env.start(); self.addCleanup(env.stop)

    def send(self, mode, description, audio=True, strength=.25, vector=None):
        ref = {'reference_mode': mode, 'audio': wav_bytes() if audio else None}
        params = Parameters().build(emotion_text=description, emo_alpha=strength, emotion_vector=vector)[0]
        result = Generate().run('正文只用于朗读，不用于自动分析情绪。', seed=43, reference=ref, parameters=params)[0]
        self.assertEqual(result['sample_rate'], 24000)
        self.assertEqual(tuple(result['waveform'].shape), (1, 1, 4800))
        self.assertEqual(params['emotion_text'], description)
        return self.requests[-1]

    def test_voice_only_uses_parameter_description(self):
        r = self.send(VOICE_ONLY, '害怕、紧张。')
        self.assertEqual(r['emotion_mode'], 'text')
        self.assertEqual(r['emotion_text'], '害怕、紧张。')
        self.assertEqual(r['emo_alpha'], .25)
        self.assertNotIn('emotion_audio', r)
        self.assertNotIn('reference_mode', r)
        self.assertEqual(base64.b64decode(r['reference_audio']), wav_bytes())

    def test_same_audio_controls_voice_and_emotion_and_discards_description(self):
        marker = '这段参数描述必须失效' * 400
        r = self.send(VOICE_AND_EMOTION, marker, vector='ignored invalid vector')
        self.assertEqual(r['emotion_mode'], 'audio')
        self.assertEqual(r['reference_audio'], r['emotion_audio'])
        self.assertEqual(r['emotion_text'], '')
        self.assertEqual(r['emo_alpha'], 1.0)
        self.assertNotIn('emotion_vector', r)
        self.assertNotIn(marker, json.dumps(r, ensure_ascii=False))

    def test_blank_description_never_falls_back_to_body(self):
        for description in ['', '   \n']:
            with self.subTest(description=description):
                r = self.send(VOICE_ONLY, description)
                self.assertEqual(r['emotion_mode'], 'vector')
                self.assertEqual(r['emotion_text'], '')
                self.assertEqual(r['emotion_vector'], [0.0] * 8)

    def test_empty_description_uses_connected_vector_in_native_order(self):
        vector = EmotionVector().build(happy=.1, angry=.8, sad=.2, afraid=.3,
                                       disgusted=.4, melancholic=.5, surprised=.6, calm=.7)[0]
        r = self.send(VOICE_ONLY, ' \n', vector=vector)
        self.assertEqual(r['emotion_mode'], 'vector')
        self.assertEqual(r['emotion_vector'], [.1, .8, .2, .3, .4, .5, .6, .7])

    def test_description_overrides_vector_without_modifying_the_description(self):
        r = self.send(VOICE_ONLY, '愤怒。', vector='ignored invalid vector')
        self.assertEqual(r['emotion_mode'], 'text')
        self.assertEqual(r['emotion_text'], '愤怒。')
        self.assertNotIn('emotion_vector', r)

    def test_empty_audio_preserves_mode_and_uses_server_default(self):
        r = self.send(VOICE_AND_EMOTION, '愤怒。', audio=False)
        self.assertEqual(r['emotion_mode'], 'reference')
        self.assertEqual(r['emotion_text'], '')
        self.assertNotIn('reference_audio', r)
        self.assertNotIn('emotion_audio', r)

    def test_disconnected_reference_uses_default_audio_and_parameter_text(self):
        Generate().run('普通正文。', parameters=Parameters().build(emotion_text='悲伤。')[0])
        self.assertEqual(self.requests[-1]['emotion_mode'], 'text')
        self.assertEqual(self.requests[-1]['emotion_text'], '悲伤。')
        self.assertNotIn('reference_audio', self.requests[-1])

    def test_upstream_has_priority_and_empty_reference_retains_mode(self):
        audio = {'waveform': torch.ones((1, 1, 4800)) * .1, 'sample_rate': 24000}
        record = Reference().bind(reference_mode=VOICE_AND_EMOTION, audio='missing.wav', audio_in=audio)[0]
        self.assertIsInstance(record['audio'], bytes)
        self.assertEqual(record['reference_mode'], VOICE_AND_EMOTION)
        self.assertTrue(Reference.VALIDATE_INPUTS(audio='missing.wav', input_types={'audio_in': 'AUDIO'}))
        self.assertEqual(Reference().bind(reference_mode=VOICE_AND_EMOTION)[0], {'reference_mode': VOICE_AND_EMOTION, 'audio': None})
        with self.assertRaisesRegex(ValueError, '上游'): Reference().bind(audio_in=None)

    def test_audio_longer_than_15_seconds_is_rejected_before_http(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'long.wav').write_bytes(wav_bytes(15.1))
            with patch.dict(sys.modules, folder_paths=types.SimpleNamespace(get_input_directory=lambda: temp)):
                with self.assertRaisesRegex(ValueError, '15 秒'): Reference().bind(audio='long.wav')
        audio = {'waveform': torch.zeros((1, 1, 15 * 24000 + 1)), 'sample_rate': 24000}
        with self.assertRaisesRegex(ValueError, '15 秒'): Reference().bind(audio_in=audio)

    def test_workflow_has_one_reference_and_no_obsolete_emotion_port(self):
        files = list((PACKAGE / 'workflows').glob('*.json'))
        self.assertEqual(len(files), 1)
        graph = json.loads(files[0].read_text())
        refs = [n for n in graph['nodes'] if n['type'] == 'IndexTTSReference']
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]['widgets_values'], [VOICE_ONLY, ''])
        gen = next(n for n in graph['nodes'] if n['type'] == 'IndexTTSGenerate')
        self.assertEqual([p['name'] for p in gen['inputs']], ['text', 'reference', 'parameters'])
        self.assertNotIn('emotion_mode', Parameters.INPUT_TYPES()['required'])
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, folder_paths=types.SimpleNamespace(get_input_directory=lambda: temp)):
                self.assertNotIn('text', Reference.INPUT_TYPES()['optional'])


if __name__ == '__main__': unittest.main()
