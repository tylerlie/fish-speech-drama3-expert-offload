"""Installation health check; --generate queues the empty-reference TTS template."""
import argparse,json,os,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fish_speech_local.core import request_api


def comfy(base,path,body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(base.rstrip('/')+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=30) as response:return json.load(response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fish',default=os.getenv('FISH_API_URL','http://127.0.0.1:18080'))
    parser.add_argument('--comfy',default='http://127.0.0.1:8188')
    parser.add_argument('--generate',action='store_true');parser.add_argument('--timeout',type=int,default=7200)
    args=parser.parse_args()
    if json.loads(request_api('/v1/health',base_url=args.fish)).get('status')!='ok':raise RuntimeError('Fish 未就绪')
    registered=comfy(args.comfy,'/object_info')
    for name in ('DramaReference','DramaReferences','DramaAPIParameters','DramaSingleVoice','DramaMultiVoice','DramaLoopReference','PrimitiveStringMultiline','PreviewAudio','SaveAudioAdvanced'):
        if name not in registered:raise RuntimeError('缺少节点：'+name)
    stats=comfy(args.comfy,'/system_stats')
    if not stats.get('devices') or any(d['type']!='cpu' for d in stats['devices']):raise RuntimeError('ComfyUI 未报告纯 CPU')
    print('Fish、六个插件节点、原生输出与 ComfyUI CPU 检查通过。')
    if not args.generate:return
    sys.path.insert(0,str(Path(__file__).parent))
    from workflow_utils import to_prompt
    graph=json.loads((Path(__file__).resolve().parents[1]/'workflows/01_单音色克隆或生成.json').read_text())
    queued=comfy(args.comfy,'/prompt',{'prompt':to_prompt(graph)})
    if queued.get('node_errors'):raise RuntimeError(queued)
    prompt_id=queued['prompt_id'];deadline=time.monotonic()+args.timeout
    print('已提交空参考工作流，等待实际生成……',flush=True)
    while time.monotonic()<deadline:
        history=comfy(args.comfy,'/history/'+prompt_id)
        if prompt_id in history:
            item=history[prompt_id]
            if item['status']['status_str']!='success':raise RuntimeError(item['status'])
            print('空参考普通生成、原生试听与保存通过。')
            print(json.dumps(item['outputs'],ensure_ascii=False,indent=2));return
        time.sleep(2)
    raise TimeoutError('等待超时；Fish 推理不一定已停止。')


if __name__=='__main__':
    try:main()
    except Exception as error:print('验证失败：'+str(error),file=sys.stderr);sys.exit(1)
