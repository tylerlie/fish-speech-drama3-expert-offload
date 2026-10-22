import base64,io,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fish_speech_local.nodes import Reference,References,SingleVoice,MultiVoice,NODE_CLASS_MAPPINGS
from fish_speech_local.audio import encode_reference,decode_audio,available_reference_media
from fish_speech_local.core import build_payload,dialogue_text,request_api
from helpers import MockFish,wav_bytes


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.file=self.root/'voice.wav';self.file.write_bytes(wav_bytes())
        self.patch=patch('fish_speech_local.audio.input_directory',return_value=self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_empty_defaults_and_whitespace(self):
        self.assertTrue(Reference.VALIDATE_INPUTS());self.assertEqual(Reference().bind(),(None,))
        self.assertEqual(Reference().bind(' ','\n'),(None,));self.assertEqual(Reference.IS_CHANGED(),'empty')
    def test_half_filled_and_missing_file(self):
        for audio,text,error in [('voice.wav','','缺少参考原文'),('','原文。','缺少参考音频'),('missing.wav','原文。','不存在')]:
            with self.subTest(audio=audio,text=text):
                self.assertIn(error,Reference.VALIDATE_INPUTS(audio,text))
                with self.assertRaisesRegex(ValueError,error):Reference().bind(audio,text)
    def test_reference_binds_file_and_transcript(self):
        record,=Reference().bind('voice.wav','实际原文。');self.assertEqual(record['text'],'实际原文。')
        samples,rate=sf.read(io.BytesIO(record['wav']));self.assertEqual(rate,22050);self.assertEqual(len(samples),4410)
    def test_cache_hash_detects_overwritten_file(self):
        old=Reference.IS_CHANGED('voice.wav','原文。');self.file.write_bytes(wav_bytes(16000))
        self.assertNotEqual(old,Reference.IS_CHANGED('voice.wav','原文。'))
    def test_input_boundary(self):
        for filename in ('../outside.wav','/etc/passwd'):
            with self.subTest(filename=filename),self.assertRaises(ValueError):Reference().bind(filename,'原文。')
    def test_corrupt_short_and_nonfinite_audio(self):
        self.file.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'无法解码'):Reference().bind('voice.wav','原文。')
        sf.write(self.file,np.zeros(100),24000)
        with self.assertRaisesRegex(ValueError,'0.1 秒'):Reference().bind('voice.wav','原文。')
        sf.write(self.file,np.full(4800,np.nan),24000,subtype='FLOAT')
        with self.assertRaisesRegex(ValueError,'无效采样'):Reference().bind('voice.wav','原文。')
    def test_stereo_encoding_and_cpu_output(self):
        sf.write(self.file,np.tile([[.2,-.2]],(4800,1)),24000,subtype='FLOAT')
        output=decode_audio(encode_reference(self.file));self.assertEqual(tuple(output['waveform'].shape),(1,1,4800))
        self.assertEqual(output['waveform'].device.type,'cpu');self.assertLess(float(output['waveform'].abs().max()),1e-5)

    def test_m4a_fallback_uses_cpu_decoder(self):
        import av
        filename=self.root/'voice.m4a'
        with av.open(str(filename),'w') as container:
            stream=container.add_stream('aac',rate=24000);stream.layout='mono'
            samples=(.1*np.sin(2*np.pi*440*np.arange(4800)/24000)).astype('float32')[None,:]
            frame=av.AudioFrame.from_ndarray(samples,format='fltp',layout='mono');frame.sample_rate=24000
            for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode(None):container.mux(packet)
        record,=Reference().bind('voice.m4a','录音原文。')
        decoded=decode_audio(record['wav'])
        self.assertEqual(decoded['sample_rate'],24000)
        self.assertGreater(decoded['waveform'].shape[-1],2400)

    def test_audio_input_port_is_optional(self):
        self.assertEqual(Reference.INPUT_TYPES()['optional']['audio_in'][0], 'AUDIO')

    def test_media_dropdown_lists_input_and_nested_files(self):
        folder = self.root/'fish_reference'; folder.mkdir()
        (folder/'旧录音.FLAC').write_bytes(wav_bytes())
        (self.root/'clip.webm').write_bytes(b'media fixture')
        (self.root/'notes.txt').write_text('not audio')
        (self.root/'.hidden.wav').write_bytes(wav_bytes())
        hidden = self.root/'.cache'; hidden.mkdir(); (hidden/'hidden.wav').write_bytes(wav_bytes())
        choices, options = Reference.INPUT_TYPES()['optional']['audio']
        self.assertEqual(choices, ['', 'clip.webm', 'fish_reference/旧录音.FLAC', 'voice.wav'])
        self.assertEqual(options['default'], '')

    def test_media_dropdown_refresh_detects_new_and_deleted_files(self):
        self.assertEqual(available_reference_media(), ['', 'voice.wav'])
        (self.root/'new.mp3').write_bytes(b'new fixture')
        self.assertIn('new.mp3', available_reference_media())
        self.file.unlink()
        self.assertNotIn('voice.wav', available_reference_media())

    def test_media_dropdown_does_not_follow_external_symlinks(self):
        with tempfile.TemporaryDirectory() as external:
            path = Path(external)/'outside.wav'; path.write_bytes(wav_bytes())
            (self.root/'linked.wav').symlink_to(path)
            (self.root/'linked-folder').symlink_to(Path(external), target_is_directory=True)
            self.assertEqual(available_reference_media(), ['', 'voice.wav'])

    def test_media_dropdown_keeps_empty_reference_without_files(self):
        self.file.unlink()
        self.assertEqual(Reference.INPUT_TYPES()['optional']['audio'][0], [''])
        self.assertEqual(Reference().bind(), (None,))

    def test_upstream_takes_priority_over_local_file(self):
        upstream = decode_audio(wav_bytes(16000))
        for local in ('', 'voice.wav', 'missing.wav', '../outside.wav'):
            with self.subTest(local=local):
                self.assertIs(Reference.VALIDATE_INPUTS(local, '上游原文。', input_types={'audio_in': 'AUDIO'}), True)
                record, = Reference().bind(local, '上游原文。', upstream)
                samples, rate = sf.read(io.BytesIO(record['wav']))
                self.assertEqual(rate, 16000)
                self.assertEqual(len(samples), 3200)
                self.assertEqual(record['text'], '上游原文。')

    def test_upstream_stereo_matches_file_encoding_without_mutation(self):
        import torch
        upstream = {'waveform': torch.tensor([[[.2]*4800, [.4]*4800]]), 'sample_rate': 24000}
        original = upstream['waveform'].clone()
        record, = Reference().bind(text='原文。', audio_in=upstream)
        sf.write(self.file, upstream['waveform'][0].numpy().T, 24000, subtype='FLOAT')
        self.assertEqual(record['wav'], encode_reference(self.file))
        self.assertTrue(torch.equal(original, upstream['waveform']))

    def test_upstream_requires_transcript(self):
        upstream = decode_audio(wav_bytes())
        self.assertIn('缺少参考原文', Reference.VALIDATE_INPUTS(input_types={'audio_in': 'AUDIO'}))
        with self.assertRaisesRegex(ValueError, '缺少参考原文'):
            Reference().bind(audio_in=upstream)

    def test_empty_upstream_never_falls_back_to_local(self):
        for upstream in (None, {}, {'waveform': None, 'sample_rate': 24000}):
            with self.subTest(upstream=upstream), self.assertRaises(ValueError):
                Reference().bind('voice.wav', '原文。', upstream)

    def test_upstream_invalid_shape_rate_and_samples(self):
        import torch
        cases = [
            (torch.zeros(1, 1, 4800), 0, 'sample_rate'),
            (torch.zeros(1, 1, 4800), 24000.5, 'sample_rate'),
            (torch.zeros(1, 4800), 24000, '三维张量'),
            (torch.zeros(2, 1, 4800), 24000, '批次'),
            (torch.zeros(1, 0, 4800), 24000, '声道'),
            (torch.zeros(1, 1, 100), 24000, '0.1 秒'),
            (torch.full((1, 1, 4800), float('nan')), 24000, '无效采样'),
            (torch.full((1, 1, 4800), float('inf')), 24000, '无效采样'),
            (torch.ones(1, 1, 4800, dtype=torch.int16), 24000, '浮点'),
        ]
        for waveform, rate, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                Reference().bind('voice.wav', '原文。', {'waveform': waveform, 'sample_rate': rate})

    def test_upstream_cache_does_not_read_local_file(self):
        import math
        self.assertTrue(math.isnan(Reference.IS_CHANGED('missing.wav', '原文。', None)))

    def test_link_validation_keeps_type_checks_and_defers_linked_text(self):
        self.assertIn('需要 AUDIO', Reference.VALIDATE_INPUTS(text='原文。', input_types={'audio_in': 'IMAGE'}))
        self.assertIs(Reference.VALIDATE_INPUTS(text=None, input_types={'audio_in': 'AUDIO', 'text': 'STRING'}), True)


class ContractTests(unittest.TestCase):
    def refs(self):return [{'text':t,'wav':b'a'} for t in ['小明：第一段。','小红：第二段。','小明：第三段。']]
    def test_empty_collection_and_sparse_order(self):
        self.assertEqual(References().collect(**{f'reference_{i}':None for i in range(1,10)}),([],))
        refs,=References().collect(reference_9=self.refs()[1],reference_2=self.refs()[0])
        self.assertEqual([r['text'] for r in refs],['小明：第一段。','小红：第二段。'])
    def test_collection_limits_and_immutability(self):
        record=self.refs()[0];refs,=References().collect(reference_1=record);refs[0]['text']='changed'
        self.assertEqual(record['text'],'小明：第一段。')
        with self.assertRaises(ValueError):References().collect(reference_10=record)
        with self.assertRaises(ValueError):References().collect(reference_1=[])
        self.assertEqual(len(NODE_CLASS_MAPPINGS),6)
    def test_single_and_multi_empty_rules(self):
        self.assertEqual(build_payload('正文。',[])['references'],[])
        with self.assertRaisesRegex(ValueError,'至少需要一个'):build_payload('小明：你好。',[],True)
    def test_plain_text_and_role_grouping(self):
        plain=build_payload('时间：12:30',[{'text':'原文：有冒号。','wav':b'a'}])
        self.assertEqual(plain['text'],'时间：12:30');self.assertEqual(plain['references'][0]['text'],'原文：有冒号。')
        multi=build_payload('小红：你好。\n小明：你好。',self.refs(),True)
        self.assertEqual([r['speaker'] for r in multi['references']],[0,1,0]);self.assertTrue(multi['text'].startswith('<|speaker:1|>'))
    def test_nine_roles_and_limit(self):
        refs=[{'text':f'角色{i}：原文。','wav':b'a'} for i in range(9)]
        self.assertEqual(build_payload('角色8：你好。',refs,True)['references'][-1]['speaker'],8)
        with self.assertRaises(ValueError):build_payload('角色8：你好。',refs+[refs[0]],True)
    def test_colons_lines_emotions_languages(self):
        ref=[{'text':'小明:[开心]原文。\n第二行。','wav':b'a'}]
        result=build_payload('小明：[开心]你好。\n 时间：八点。\nhttps://example.com\n12:30',ref,True)
        self.assertIn('[happy]你好。',result['text']);self.assertEqual(result['text'].count('<|speaker:'),1)
        self.assertEqual(result['references'][0]['text'],'[开心]原文。\n第二行。')
        for text in ('中文。','Hello.','こんにちは。','Bonjour.'):
            self.assertEqual(build_payload(text)['text'],text)
    def test_bad_scripts_and_long_dialogue(self):
        for text in ('','小明：\n小红：你好。','<|speaker:0|>你好。'):
            with self.subTest(text=text),self.assertRaises(ValueError):build_payload(text,self.refs(),True)
        with self.assertRaisesRegex(ValueError,'第 4 行.*小王'):dialogue_text('\n\n小明：你好。\n小王：你好。',['小明'])
        p=build_payload('小明：中文。Hello.\n小红：こんにちは。Bonjour.\n'*100,self.refs(),True)
        self.assertEqual(p['text'].count('<|speaker:'),200);self.assertFalse(p['single_chunk'])


class TransportTests(unittest.TestCase):
    def test_http_payload_and_cpu_audio(self):
        with MockFish() as api:
            output,=MultiVoice().run('小明：你好。',references=[{'text':'小明：原文。','wav':wav_bytes()}])
            self.assertEqual(output['sample_rate'],22050);self.assertEqual(output['waveform'].device.type,'cpu')
            sent=api.requests[0][1];self.assertEqual(sent['references'][0]['text'],'原文。')
            self.assertTrue(base64.b64decode(sent['references'][0]['audio']).startswith(b'RIFF'))
    def test_failure_no_retry_and_bad_response(self):
        with MockFish() as api:
            api.error=True
            with self.assertRaisesRegex(RuntimeError,'HTTP 500'):SingleVoice().run('你好。')
            self.assertEqual(len(api.requests),1);api.error=False;api.bad_audio=True
            with self.assertRaisesRegex(RuntimeError,'WAV'):SingleVoice().run('你好。')
    def test_proxy_bypass_optional_auth_and_health(self):
        with MockFish() as api,patch.dict(os.environ,{'HTTP_PROXY':'http://127.0.0.1:1','FISH_API_KEY':'test'}):
            self.assertIn(b'ok',request_api('/v1/health'));SingleVoice().run('你好。')
            self.assertEqual(api.requests[0][2]['Authorization'],'Bearer test')
    def test_serial_requests(self):
        with MockFish() as api,ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i:SingleVoice().run(f'第{i}句。'),range(2)));self.assertEqual(api.max_active,1)
    def test_invalid_url_timeout_and_decoding(self):
        for url in ('http://localhost:1/v1/tts','file:///tmp/a','http://localhost:bad'):
            with self.subTest(url=url),self.assertRaises(ValueError):request_api('/v1/health',base_url=url)
        with patch.dict(os.environ,{'FISH_REQUEST_TIMEOUT':'0'}),self.assertRaises(ValueError):SingleVoice().run('你好。')
        with self.assertRaises(RuntimeError):decode_audio(b'broken')


if __name__=='__main__':unittest.main(verbosity=2)
