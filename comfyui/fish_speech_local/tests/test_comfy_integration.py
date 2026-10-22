"""Actual ComfyUI CPU execution against MockFish. Opt in with COMFY_SOURCE.

COMFY_SOURCE=/path/to/ComfyUI python -m unittest discover -s tests -v
"""
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.parse
import urllib.request

import soundfile as sf

from helpers import MockFish, wav_bytes

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'tools'))
from workflow_utils import to_prompt


@unittest.skipUnless(os.getenv('COMFY_SOURCE'), 'Set COMFY_SOURCE to test actual ComfyUI CPU execution.')
class ActualComfyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='fish-comfy-acceptance-')
        cls.base = Path(cls.temporary.name)
        for name in ('custom_nodes', 'input', 'output', 'user', 'cache'):
            (cls.base / name).mkdir()
        (cls.base / 'custom_nodes/fish_speech_local').symlink_to(PACKAGE, target_is_directory=True)
        if os.getenv('LOAD_BY_URL_SOURCE'):
            (cls.base / 'custom_nodes/load_by_url').symlink_to(
                Path(os.environ['LOAD_BY_URL_SOURCE']).resolve(), target_is_directory=True)
        for name, rate in [('reference-a.wav', 24000), ('reference-b.wav', 16000)]:
            (cls.base / 'input' / name).write_bytes(wav_bytes(rate))
        cls.fish = MockFish().__enter__()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        cls.url = f'http://127.0.0.1:{port}'
        cls.log_path = cls.base / 'comfy.log'
        cls.log = cls.log_path.open('w')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONUNBUFFERED': '1',
               'HF_HOME': str(cls.base / 'cache/hf'), 'XDG_CACHE_HOME': str(cls.base / 'cache'),
               'TORCH_HOME': str(cls.base / 'cache/torch'), 'MPLCONFIGDIR': str(cls.base / 'cache/mpl')}
        cls.proc = subprocess.Popen([sys.executable, 'main.py', '--cpu', '--listen', '127.0.0.1',
            '--port', str(port), '--disable-partner-nodes', '--disable-auto-launch',
            '--base-directory', str(cls.base), '--user-directory', str(cls.base / 'user')],
            cwd=os.environ['COMFY_SOURCE'], env=env, stdout=cls.log, stderr=cls.log)
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                if cls.proc.poll() is not None:
                    raise RuntimeError(cls.log_path.read_text(errors='replace'))
                try:
                    registered = cls.request('/object_info')
                    if 'DramaSingleVoice' in registered:
                        cls.registered = registered
                        return
                except OSError:
                    pass
                time.sleep(0.5)
            raise TimeoutError(cls.log_path.read_text(errors='replace'))
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                cls.proc.kill(); cls.proc.wait()
        cls.log.close()
        cls.fish.__exit__()
        cls.temporary.cleanup()

    @classmethod
    def request(cls, path, body=None, raw=False):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        req = urllib.request.Request(cls.url + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10) as response:
            content = response.read()
            return content if raw else json.loads(content)

    def setUp(self):
        self.before = len(self.fish.requests)

    def graph(self, dialogue=False):
        name = '02_多角色长文本.json' if dialogue else '01_单音色克隆或生成.json'
        graph = json.loads((PACKAGE / 'workflows' / name).read_text())
        # Existing acceptance cases cover the original one-request behavior.
        if not dialogue:
            next(n for n in graph['nodes'] if n['type']=='DramaSingleVoice')['widgets_values'][2]=0
        return graph

    def set_ref(self,graph,slot=1,text='小明：原文。',audio='reference-a.wav'):
        [n for n in graph['nodes'] if n['type']=='DramaReference'][slot-1]['widgets_values']=[audio,text]
    def set_text(self,graph,text):
        next(n for n in graph['nodes'] if n['type']=='PrimitiveStringMultiline')['widgets_values'][0]=text
    def execute(self,graph,success=True):
        return self.execute_prompt(to_prompt(graph), success)
    def execute_prompt(self,prompt,success=True):
        queued=self.request('/prompt',{'prompt':prompt});self.assertFalse(queued.get('node_errors'),queued)
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            result=self.request('/history/'+queued['prompt_id'])
            if queued['prompt_id'] in result:
                item=result[queued['prompt_id']]
                self.assertEqual(item['status']['status_str'],'success' if success else 'error',item['status']);return item
            time.sleep(.1)
        self.fail('Comfy timeout')
    def assert_audio(self,graph,item):
        for kind in ('PreviewAudio','SaveAudioAdvanced'):
            number=next(n['id'] for n in graph['nodes'] if n['type']==kind)
            entry=item['outputs'][str(number)]['audio'][0]
            data=self.request('/view?'+urllib.parse.urlencode(entry),raw=True)
            samples,rate=sf.read(io.BytesIO(data));self.assertEqual(rate,22050);self.assertEqual(len(samples),4410)
    def rejected(self,graph,message):
        try:self.request('/prompt',{'prompt':to_prompt(graph)})
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code,400);self.assertIn(message,json.dumps(json.loads(error.read()),ensure_ascii=False))
        else:self.fail('Expected HTTP 400')
        self.assertEqual(len(self.fish.requests),self.before)
    def test_01_single_empty_connected_reference(self):
        g=self.graph();self.assert_audio(g,self.execute(g));self.assertEqual(self.fish.requests[-1][1]['references'],[])
    def test_02_single_clone_same_wiring(self):
        g=self.graph();self.set_ref(g,text='实际原文。');self.assert_audio(g,self.execute(g))
        self.assertEqual(self.fish.requests[-1][1]['references'][0]['text'],'实际原文。')
    def test_03_only_first_of_nine_filled(self):
        g=self.graph(True);self.set_ref(g);self.set_text(g,'小明：你好。');self.assert_audio(g,self.execute(g))
        self.assertEqual(len(self.fish.requests[-1][1]['references']),1)
    def test_04_only_ninth_filled(self):
        g=self.graph(True);self.set_ref(g,9,'小红：原文。');self.set_text(g,'小红：你好。');self.execute(g)
        self.assertEqual(self.fish.requests[-1][1]['references'][0]['speaker'],0)
    def test_05_nonconsecutive_roles(self):
        g=self.graph(True);self.set_ref(g,2);self.set_ref(g,8,'小红：原文。','reference-b.wav');self.execute(g)
        self.assertEqual([r['speaker'] for r in self.fish.requests[-1][1]['references']],[0,1])
    def test_06_all_nine_roles(self):
        g=self.graph(True)
        for i in range(1,10):self.set_ref(g,i,f'角色{i}：原文。')
        self.set_text(g,'\n'.join(f'角色{i}：你好。' for i in range(1,10)));self.execute(g)
        self.assertEqual([r['speaker'] for r in self.fish.requests[-1][1]['references']],list(range(9)))
    def test_07_empty_multi_is_error(self):
        item=self.execute(self.graph(True),False);self.assertEqual(len(self.fish.requests),self.before)
        self.assertIn('至少需要一个',json.dumps(item['status'],ensure_ascii=False))
    def test_08_audio_only(self):
        g=self.graph();self.set_ref(g,text='');self.rejected(g,'缺少参考原文')
    def test_09_text_only(self):
        g=self.graph();self.set_ref(g,audio='');self.rejected(g,'缺少参考音频')
    def test_10_missing_file(self):
        g=self.graph();self.set_ref(g,audio='missing.wav');self.rejected(g,'不存在')
    def test_11_unknown_role_no_api(self):
        g=self.graph(True);self.set_ref(g);self.set_text(g,'小王：你好。');self.execute(g,False)
        self.assertEqual(len(self.fish.requests),self.before)
    def test_12_repeat_requests(self):
        g=self.graph();self.execute(g);self.execute(g);self.assertEqual(len(self.fish.requests)-self.before,2)
    def test_13_same_role_multiple_clips(self):
        g=self.graph(True);self.set_ref(g);self.set_ref(g,9,'小明：另一段。');self.set_text(g,'小明：你好。');self.execute(g)
        self.assertEqual([r['speaker'] for r in self.fish.requests[-1][1]['references']],[0,0])
    def test_14_long_multilingual_one_request(self):
        g=self.graph(True);self.set_ref(g);self.set_text(g,'小明：中文。Hello. こんにちは。Bonjour.\n'*200);self.execute(g)
        self.assertEqual(len(self.fish.requests)-self.before,1)
    def test_15_cpu_native_nodes_and_extension(self):
        own={n for n in self.registered if n.startswith(('Drama','FishSpeech','FishLocal'))}
        self.assertEqual(own,{'DramaReference','DramaReferences','DramaAPIParameters','DramaSingleVoice','DramaMultiVoice','DramaLoopReference'})
        self.assertEqual(len(self.request('/workflow_templates')['fish_speech_local']),3)
        self.assertTrue(all(d['type']=='cpu' for d in self.request('/system_stats')['devices']))
        self.assertIn('上传参考音频'.encode(),self.request('/extensions/fish_speech_local/reference.js',raw=True))
    def test_16_upload_and_playback_endpoint(self):
        boundary='TestFish2026'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\nfish_reference\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="uploaded.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+wav_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        req=urllib.request.Request(self.url+'/upload/image',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=20) as response:uploaded=json.load(response)
        g=self.graph();self.set_ref(g,audio=uploaded['subfolder']+'/'+uploaded['name'],text='原文。');self.execute(g)
        audio=self.request('/view?'+urllib.parse.urlencode({'filename':uploaded['name'],'subfolder':uploaded['subfolder'],'type':'input'}),raw=True)
        self.assertEqual(audio,wav_bytes())
    def test_17_optional_reference_fields_omitted(self):
        from unittest.mock import patch
        g=self.graph();prompt=to_prompt(g)
        next(n for n in prompt.values() if n['class_type']=='DramaReference')['inputs']={}
        with patch(__name__+'.to_prompt',return_value=prompt):self.execute(g)
    def test_18_api_failure_no_retry(self):
        self.fish.error=True
        try:self.execute(self.graph(),False);self.assertEqual(len(self.fish.requests)-self.before,1)
        finally:self.fish.error=False

    def test_19_verification_script(self):
        fish='http://127.0.0.1:'+str(self.fish.server.server_port)
        result=subprocess.run([sys.executable,str(PACKAGE/'tools/verify.py'),'--fish',fish,
                               '--comfy',self.url,'--generate'],capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('空参考普通生成',result.stdout)

    def loop_graph(self, text='小明：第一句。\n小红：[开心]第二句。\n小明：[低声]第三句。'):
        graph = json.loads((PACKAGE/'workflows/03_拼接式多角色多语气.json').read_text())
        next(n for n in graph['nodes'] if n['type']=='DramaLoopReference')['widgets_values'][3]=0
        self.set_ref(graph)
        self.set_ref(graph, 9, '小红：女声参考。', 'reference-b.wav')
        self.set_text(graph, text)
        return graph

    def output_audio(self, item, number):
        outputs = item['outputs'][str(number)]['audio']
        return [sf.read(io.BytesIO(self.request('/view?'+urllib.parse.urlencode(entry), raw=True)))
                for entry in outputs]

    def test_20_loop_full_and_array_native_output_nodes(self):
        import numpy as np
        g = self.loop_graph(); item = self.execute(g)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual([r['text'] for r in sent], ['第一句。', '[happy]第二句。', '[whisper]第三句。'])
        self.assertEqual([r['references'][0]['text'] for r in sent], ['原文。', '女声参考。', '原文。'])
        self.assertEqual([r['references'][0]['speaker'] for r in sent], [0, 0, 0])
        self.assertEqual([r['seed'] for r in sent], [42, 43, 44])
        for kind in ('PreviewAudio', 'SaveAudioAdvanced'):
            full_node = next(n for n in g['nodes'] if n['type']==kind and n['title'].startswith('完整拼接'))
            array_node = next(n for n in g['nodes'] if n['type']==kind and n['title'].startswith('分段数组'))
            full = self.output_audio(item, full_node['id']); parts = self.output_audio(item, array_node['id'])
            self.assertEqual(len(full), 1); self.assertEqual(len(parts), 3)
            self.assertEqual(full[0][1], 22050)
            expected = np.concatenate([parts[0][0], np.zeros(4410), parts[1][0], np.zeros(4410), parts[2][0]])
            self.assertTrue(np.array_equal(full[0][0], expected))

    def test_21_loop_blank_lines_and_sparse_reference(self):
        g = self.loop_graph('\n小红：一句。\n\n小红：[开心]二句。\n')
        self.set_ref(g, audio='', text='')
        self.execute(g)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(len(r['references'])==1 and r['references'][0]['speaker']==0 for r in sent))

    def test_22_loop_all_empty_no_inference(self):
        g = self.loop_graph(); self.set_ref(g, audio='', text=''); self.set_ref(g, 9, text='', audio='')
        self.execute(g, False)
        self.assertEqual(len(self.fish.requests), self.before)

    def test_23_loop_unknown_final_line_no_partial_requests(self):
        g = self.loop_graph('小明：有效。\n小王：未绑定。')
        item = self.execute(g, False)
        self.assertIn('第 2 行', json.dumps(item['status'], ensure_ascii=False))
        self.assertEqual(len(self.fish.requests), self.before)

    def test_24_loop_middle_failure_stops(self):
        self.fish.fail_texts.add('故障句。')
        try:
            item = self.execute(self.loop_graph('小明：正常。\n小红：故障句。\n小明：不执行。'), False)
            self.assertEqual(len(self.fish.requests)-self.before, 2)
            self.assertIn('第 2 行', json.dumps(item['status'], ensure_ascii=False))
            self.assertEqual(item['outputs'], {})
        finally:
            self.fish.fail_texts.clear()

    def test_25_loop_one_line_no_leading_trailing_gap(self):
        g = self.loop_graph('小红：单句。')
        next(n for n in g['nodes'] if n['type']=='DramaLoopReference')['widgets_values'][2] = 1000
        item = self.execute(g)
        for n in g['nodes']:
            if n['type'] in ('PreviewAudio', 'SaveAudioAdvanced'):
                output = self.output_audio(item, n['id'])
                self.assertEqual(len(output), 1); self.assertEqual(len(output[0][0]), 4410)

    def test_26_loop_nine_roles_and_zero_gap(self):
        g = self.loop_graph('\n'.join(f'角色{i}：第{i}句。' for i in range(1, 10)))
        for i in range(1, 10): self.set_ref(g, i, f'角色{i}：参考{i}。')
        next(n for n in g['nodes'] if n['type']=='DramaLoopReference')['widgets_values'][2] = 0
        item = self.execute(g)
        self.assertEqual(len(self.fish.requests)-self.before, 9)
        full_node = next(n for n in g['nodes'] if n['type']=='PreviewAudio' and n['title'].startswith('完整拼接'))
        self.assertEqual(len(self.output_audio(item, full_node['id'])[0][0]), 9*4410)

    def test_27_loop_list_type_and_repeat_execution(self):
        info = self.registered['DramaLoopReference']
        self.assertEqual(info['output'], ['AUDIO', 'AUDIO'])
        self.assertEqual(info['output_is_list'], [False, True])
        g = self.loop_graph('小明：同一行。'); self.execute(g); self.execute(g)
        self.assertEqual(len(self.fish.requests)-self.before, 2)

    def audio_input_prompt(self, local='', upstream_type='LoadAudio', upstream_inputs=None):
        graph = self.graph()
        self.set_ref(graph, audio=local, text='上游录音原文。')
        prompt = to_prompt(graph)
        ref = next(n for n in prompt.values() if n['class_type'] == 'DramaReference')
        ref['inputs']['audio_in'] = ['1000', 0]
        prompt['1000'] = {'class_type': upstream_type, 'inputs': upstream_inputs or {'audio': 'reference-b.wav'}}
        return prompt

    def test_28_native_audio_input_with_empty_local_file(self):
        self.execute_prompt(self.audio_input_prompt())
        import base64
        reference = self.fish.requests[-1][1]['references'][0]
        samples, rate = sf.read(io.BytesIO(base64.b64decode(reference['audio'])))
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 3200)
        self.assertEqual(reference['text'], '上游录音原文。')

    def test_29_native_audio_overrides_missing_local_file_and_repeats(self):
        prompt = self.audio_input_prompt('missing.wav')
        self.execute_prompt(prompt); self.execute_prompt(prompt)
        self.assertEqual(len(self.fish.requests)-self.before, 2)

    @unittest.skipUnless(os.getenv('LOAD_BY_URL_SOURCE'), 'Set LOAD_BY_URL_SOURCE to test the real URL loader.')
    def test_30_real_url_loader_trimmed_audio_overrides_local(self):
        import base64
        self.assertIn('LoadAudioByUrl', self.registered)
        prompt = self.audio_input_prompt('reference-b.wav', 'LoadAudioByUrl', {
            'url': os.environ['FISH_API_URL']+'/reference.wav', 'max_seconds': .15, 'skip_first_seconds': .025})
        self.execute_prompt(prompt)
        samples, rate = sf.read(io.BytesIO(base64.b64decode(self.fish.requests[-1][1]['references'][0]['audio'])))
        self.assertEqual(rate, 22050)
        self.assertEqual(len(samples), int(22050*.15))

    @unittest.skipUnless(os.getenv('LOAD_BY_URL_SOURCE'), 'Set LOAD_BY_URL_SOURCE to test the real URL loader.')
    def test_31_empty_url_does_not_use_local_reference(self):
        prompt = self.audio_input_prompt('reference-a.wav', 'LoadAudioByUrl', {
            'url': '', 'max_seconds': 0., 'skip_first_seconds': 0.})
        item = self.execute_prompt(prompt, False)
        self.assertIn('上游未提供有效 AUDIO', json.dumps(item['status'], ensure_ascii=False))
        self.assertEqual(len(self.fish.requests), self.before)

    def test_32_linked_transcript_and_audio(self):
        prompt = self.audio_input_prompt('missing.wav')
        prompt['1001'] = {'class_type': 'PrimitiveStringMultiline', 'inputs': {'value': '连线原文。'}}
        next(n for n in prompt.values() if n['class_type'] == 'DramaReference')['inputs']['text'] = ['1001', 0]
        self.execute_prompt(prompt)
        self.assertEqual(self.fish.requests[-1][1]['references'][0]['text'], '连线原文。')

    def set_parameters(self, graph, **values):
        from fish_speech_local.parameters import PARAMETER_INPUTS
        node = next(n for n in graph['nodes'] if n['type']=='DramaAPIParameters')
        for name, value in values.items():
            node['widgets_values'][list(PARAMETER_INPUTS).index(name)] = value

    def test_33_all_templates_include_connected_parameters(self):
        for g in (self.graph(), self.graph(True), self.loop_graph()):
            prompt = to_prompt(g)
            gen = next(n for n in prompt.values() if n['class_type'] in ('DramaSingleVoice','DramaMultiVoice','DramaLoopReference'))
            source = prompt[gen['inputs']['parameters'][0]]
            self.assertEqual(source['class_type'], 'DramaAPIParameters')
            self.assertEqual(source['inputs']['temperature'], 1.)
            self.assertEqual(source['inputs']['top_p'], .9)

    def test_34_all_three_modes_transmit_configuration(self):
        graphs = [self.graph(), self.graph(True), self.loop_graph('小明：一。\n小红：二。')]
        self.set_ref(graphs[1]); self.set_text(graphs[1], '小明：多角色。')
        for g in graphs:
            self.set_parameters(g, temperature=.85, top_p=.95, max_new_tokens=600,
                seed_mode='参数节点种子', seed=99, single_chunk=True,
                use_memory_cache='on', chunk_length=450, repetition_penalty=1.5, normalize=False, latency='balanced')
            self.execute(g)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual([p['seed'] for p in sent], [99, 99, 99, 100])
        for p in sent:
            self.assertEqual((p['temperature'], p['top_p'], p['max_new_tokens']), (.85, .95, 600))
            self.assertTrue(p['single_chunk']); self.assertFalse(p['normalize'])
            self.assertEqual(p['latency'], 'balanced')
            self.assertEqual(p['chunk_length'], 450)

    def test_35_parameter_change_reaches_repeated_request(self):
        g = self.graph(); self.execute(g)
        self.set_parameters(g, temperature=.6, top_p=.8)
        self.execute(g)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual([p['temperature'] for p in sent], [1., .6])

    def test_36_old_prompts_without_parameter_connection(self):
        graphs = [self.graph(), self.graph(True), self.loop_graph('小明：一句。')]
        self.set_ref(graphs[1]); self.set_text(graphs[1], '小明：多角色。')
        for g in graphs:
            prompt = to_prompt(g)
            number = next(k for k,n in prompt.items() if n['class_type']=='DramaAPIParameters')
            prompt.pop(number)
            for n in prompt.values(): n['inputs'].pop('parameters', None)
            self.execute_prompt(prompt)
        self.assertTrue(all('temperature' not in r[1] and 'top_p' not in r[1]
                            for r in self.fish.requests[self.before:]))

    def test_37_each_response_format_with_native_preview_and_save(self):
        for fmt in ('wav', 'mp3', 'opus', 'pcm'):
            with self.subTest(fmt=fmt):
                g = self.graph(); self.set_parameters(g, format=fmt)
                item = self.execute(g)
                saved = next(n for n in g['nodes'] if n['type']=='SaveAudioAdvanced')
                samples, rate = self.output_audio(item, saved['id'])[0]
                self.assertGreater(len(samples), 1000)
                self.assertGreater(rate, 0)
                self.assertEqual(self.fish.requests[-1][1]['format'], fmt)

    def test_38_streaming_wav_and_no_seed_modes(self):
        g = self.loop_graph('小明：一句。\n小红：二句。')
        self.set_parameters(g, streaming=True, seed_mode='不指定种子')
        self.execute(g)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(p['streaming'] and 'seed' not in p for p in sent))

    def test_39_invalid_streaming_combination_before_fish_request(self):
        g = self.graph(); self.set_parameters(g, format='mp3', streaming=True)
        item = self.execute(g, False)
        self.assertIn('只支持 WAV', json.dumps(item['status'], ensure_ascii=False))
        self.assertEqual(len(self.fish.requests), self.before)

    def test_40_same_parameter_node_fans_out_to_three_generators(self):
        from fish_speech_local.parameters import PARAMETER_INPUTS
        values = next(n for n in self.graph()['nodes'] if n['type']=='DramaAPIParameters')['widgets_values']
        inputs = dict(zip(PARAMETER_INPUTS, values))
        inputs.update(seed_mode='参数节点种子', seed=70)
        prompt = {'1': {'class_type': 'DramaAPIParameters', 'inputs': inputs},
            '2': {'class_type': 'DramaReference', 'inputs': {'audio': 'reference-a.wav', 'text': '小明：原文。'}},
            '3': {'class_type': 'DramaReferences', 'inputs': {'reference_1': ['2', 0]}}}
        for number, kind, text in [('4','DramaSingleVoice','单音色。'),
                                   ('5','DramaMultiVoice','小明：多角色。'),
                                   ('6','DramaLoopReference','小明：一。\n小明：二。')]:
            prompt[number] = {'class_type': kind, 'inputs': {
                'text': text, 'seed': 42, 'references': ['3',0], 'parameters': ['1',0]}}
            if kind=='DramaLoopReference': prompt[number]['inputs']['gap_ms'] = 0
            prompt['10'+number] = {'class_type':'PreviewAudio', 'inputs':{'audio':[number,0]}}
        self.execute_prompt(prompt)
        sent = [r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual(len(sent), 4)
        self.assertEqual(sorted(p['seed'] for p in sent), [70,70,70,71])

    def test_41_saved_reference_id_without_local_reference(self):
        g = self.graph(); self.set_parameters(g, reference_id='saved_voice-1')
        self.execute(g)
        self.assertEqual(self.fish.requests[-1][1]['reference_id'], 'saved_voice-1')
        self.assertEqual(self.fish.requests[-1][1]['references'], [])

    def test_42_reference_media_combo_lists_existing_files(self):
        info = self.request('/object_info/DramaReference')['DramaReference']
        choices, options = info['input']['optional']['audio']
        self.assertEqual(choices[0], '')
        self.assertIn('reference-a.wav', choices)
        self.assertIn('reference-b.wav', choices)
        self.assertEqual(options['default'], '')

    def test_43_reuse_nested_media_without_upload(self):
        folder = self.base/'input/fish_reference'; folder.mkdir(exist_ok=True)
        existing = folder/'已上传录音.wav'; existing.write_bytes(wav_bytes(16000))
        try:
            choices = self.request('/object_info/DramaReference')['DramaReference']['input']['optional']['audio'][0]
            self.assertIn('fish_reference/已上传录音.wav', choices)
            g = self.graph(); self.set_ref(g, audio='fish_reference/已上传录音.wav', text='实际原文。')
            self.execute(g)
            import base64
            received = base64.b64decode(self.fish.requests[-1][1]['references'][0]['audio'])
            samples, rate = sf.read(io.BytesIO(received))
            self.assertEqual(rate, 16000); self.assertEqual(len(samples), 3200)
        finally: existing.unlink()

    def test_44_media_refresh_removes_deleted_files_without_resetting_execution_rules(self):
        path = self.base/'input/added.wav'; path.write_bytes(wav_bytes())
        choices = lambda: self.request('/object_info/DramaReference')['DramaReference']['input']['optional']['audio'][0]
        self.assertIn('added.wav', choices())
        path.unlink()
        self.assertNotIn('added.wav', choices())
        g = self.graph(); self.set_ref(g, audio='added.wav', text='原文。')
        self.rejected(g, '不存在')

    def test_45_default_single_template_performs_four_regenerations(self):
        g=json.loads((PACKAGE/'workflows/01_单音色克隆或生成.json').read_text())
        self.set_ref(g,text='原始录音原文。');self.set_text(g,'[惊喜]你回来了！')
        item=self.execute(g);self.assert_audio(g,item)
        sent=[r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual(len(sent),5)
        self.assertEqual([len(p['references']) for p in sent],[1,2,2,2,2])
        self.assertEqual([p['seed'] for p in sent],[42,1042,2042,3042,3042])
        self.assertTrue(all(p['single_chunk'] for p in sent))
        self.assertTrue(all(p['references'][-1]['text']=='你回来了！' for p in sent[1:]))

    def test_46_default_loop_template_keeps_role_anchors_and_final_segment_count(self):
        g=json.loads((PACKAGE/'workflows/03_拼接式多角色多语气.json').read_text())
        self.set_ref(g,text='小明：男声原文。')
        self.set_ref(g,9,text='小红：女声原文。',audio='reference-b.wav')
        self.set_text(g,'小明：[惊喜]你好。\n小红：[轻声]再见。')
        item=self.execute(g)
        sent=[r[1] for r in self.fish.requests[self.before:]]
        self.assertEqual(len(sent),10)
        self.assertEqual([len(p['references']) for p in sent],[1,2,2,2,2]*2)
        self.assertTrue(all(p['references'][0]['text']=='男声原文。' for p in sent[:5]))
        self.assertTrue(all(p['references'][0]['text']=='女声原文。' for p in sent[5:]))
        output=next(n for n in g['nodes'] if n['type']=='PreviewAudio' and n['title'].startswith('分段数组'))
        self.assertEqual(len(self.output_audio(item,output['id'])),2)


if __name__=='__main__':unittest.main(verbosity=2)
