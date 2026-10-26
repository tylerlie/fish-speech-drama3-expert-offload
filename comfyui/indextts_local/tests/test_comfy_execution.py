"""Optional real CPU ComfyUI + deployed HTTP adapter with a recording model.

Set COMFY_SOURCE and INDEX_API_SOURCE to existing source locations; no downloads.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
import soundfile as sf

PACKAGE = Path(__file__).resolve().parents[1]


def wav_bytes():
    data = io.BytesIO()
    sf.write(data, np.sin(np.arange(4800) * .03) * .1, 24000, format='WAV')
    return data.getvalue()


@unittest.skipUnless(os.getenv('COMFY_SOURCE') and os.getenv('INDEX_API_SOURCE'),
                     'Set COMFY_SOURCE and INDEX_API_SOURCE for CPU ComfyUI integration')
class ComfyExecutionTest(unittest.TestCase):
    def test_actual_workflow_modes_default_reference_and_upstream_priority(self):
        spec = importlib.util.spec_from_file_location('index_api_under_test', os.environ['INDEX_API_SOURCE'])
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        calls = []
        class RecordingModel:
            def normalize_emo_vec(self, vector, apply_bias=True):
                return vector

            def infer(self, **kwargs):
                calls.append(kwargs)
                Path(kwargs['output_path']).write_bytes(wav_bytes())
        with tempfile.TemporaryDirectory(prefix='index-v020-') as temp:
            base = Path(temp)
            for name in ['custom_nodes','input','output','user','cache']:(base/name).mkdir()
            (base/'input/reference.wav').write_bytes(wav_bytes())
            (base/'custom_nodes/indextts_local').symlink_to(PACKAGE, target_is_directory=True)
            engine = adapter.Engine(RecordingModel(), '2.5', 'recording', base/'input/reference.wav')
            server = ThreadingHTTPServer(('127.0.0.1',0),adapter.handler_for(engine))
            thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def request(path, body=None):
                req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
                    json.dumps(body).encode() if body is not None else None, {'Content-Type':'application/json'})
                with opener.open(req,timeout=10) as response:return json.load(response)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','INDEX_TTS_API_URL':f'http://127.0.0.1:{server.server_port}',
                 'HF_HOME':str(base/'cache'),'TORCH_HOME':str(base/'cache')}
            log=base/'comfy.log'
            try:
                with log.open('w') as output:
                    proc=subprocess.Popen([sys.executable,'main.py','--cpu','--listen','127.0.0.1','--port',str(port),
                        '--disable-partner-nodes','--disable-auto-launch','--base-directory',str(base),
                        '--user-directory',str(base/'user')],cwd=os.environ['COMFY_SOURCE'],env=env,stdout=output,stderr=output)
                    try:
                        deadline=time.monotonic()+120
                        while time.monotonic()<deadline:
                            if proc.poll() is not None:self.fail(log.read_text())
                            try:
                                info=request('/object_info')
                                if 'IndexTTSGenerate' in info:break
                            except OSError:pass
                            time.sleep(.5)
                        else:self.fail(log.read_text())
                        graph=json.loads(next((PACKAGE/'workflows').glob('*.json')).read_text())
                        links={link[0]:link for link in graph['links']}
                        prompt={}
                        for node in graph['nodes']:
                            kind=node['type']; values=iter(node['widgets_values'])
                            inputs={p['name']:[str(links[p['link']][1]),links[p['link']][2]] for p in node['inputs'] if p.get('link')}
                            schema=info[kind]['input']
                            for name,specification in {**schema.get('required',{}),**schema.get('optional',{})}.items():
                                typ=specification[0];opts=specification[1] if len(specification)>1 else {}
                                if name in inputs or opts.get('forceInput'):continue
                                if isinstance(typ,list) or typ in ['STRING','INT','FLOAT','BOOLEAN']:
                                    inputs[name]=next(values)
                                    if opts.get('control_after_generate'):next(values)
                            if kind=='SaveAudioAdvanced':inputs['format']='flac'
                            prompt[str(node['id'])]={'class_type':kind,'inputs':inputs}
                        cases = [('仅参考音色','upload','参数节点的恐惧情绪。'),
                                 ('参考音色＋情绪','upload','应被音频覆盖的描述。'),
                                 ('仅参考音色','empty','参数节点的悲伤情绪。'),
                                 ('参考音色＋情绪','empty','应被音频覆盖的描述。'),
                                 ('参考音色＋情绪','upstream','应被音频覆盖的描述。'),
                                 ('仅参考音色','upload',''),
                                 ('仅参考音色','empty','  \n')]
                        for mode,audio_kind,description in cases:
                            with self.subTest(mode=mode,audio_kind=audio_kind,description=description):
                                prompt['1']['inputs'].update(reference_mode=mode,audio='reference.wav' if audio_kind=='upload' else '')
                                prompt['1']['inputs'].pop('audio_in',None)
                                prompt.pop('8',None)
                                prompt['3']['inputs']['emotion_text']=description
                                prompt['7']['inputs']['angry']=.8
                                if audio_kind=='upstream':
                                    prompt['8']={'class_type':'LoadAudio','inputs':{'audio':'reference.wav'}}
                                    prompt['1']['inputs'].update(audio='missing-local.wav',audio_in=['8',0])
                                queued=request('/prompt',{'prompt':prompt})
                                self.assertFalse(queued.get('node_errors'),queued)
                                deadline=time.monotonic()+45
                                while time.monotonic()<deadline:
                                    history=request('/history/'+queued['prompt_id'])
                                    if queued['prompt_id'] in history:
                                        item=history[queued['prompt_id']]
                                        self.assertEqual(item['status']['status_str'],'success',item)
                                        break
                                    time.sleep(.1)
                                else:self.fail(log.read_text())
                                call=calls[-1]
                                if mode=='仅参考音色' and description.strip():
                                    self.assertTrue(call['use_emo_text'])
                                    self.assertEqual(call['emo_text'],description)
                                    self.assertNotIn('emo_vector',call)
                                    self.assertIsNone(call['emo_audio_prompt'])
                                elif mode=='仅参考音色':
                                    self.assertFalse(call['use_emo_text'])
                                    self.assertIsNone(call['emo_text'])
                                    self.assertEqual(call['emo_vector'],[0,.8,0,0,0,0,0,0])
                                else:
                                    self.assertFalse(call['use_emo_text']);self.assertIsNone(call['emo_text'])
                                    self.assertNotIn('emo_vector',call)
                                    if audio_kind!='empty':self.assertEqual(call['spk_audio_prompt'],call['emo_audio_prompt'])
                        self.assertEqual(len(calls),7)
                        self.assertEqual(len(list((base/'output').rglob('*.flac'))),7)
                    finally:
                        proc.terminate()
                        try:proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
            finally:
                server.shutdown();server.server_close();thread.join();engine.audio_cache.cleanup()


if __name__=='__main__':unittest.main()
