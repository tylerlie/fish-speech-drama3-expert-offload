"""Build the single IndexTTS 2.0/2.5 workflow template."""
import json
import sys
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from indextts_local.nodes import PARAMETERS, VOICE_ONLY, EMOTIONS

class Workflow:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []

    def node(self, node_type, title, pos, inputs=None, outputs=None, widgets=None, size=None):
        number = len(self.nodes) + 1
        item = {'id': number, 'type': node_type, 'title': title, 'pos': pos,
                'size': size or [330, 180], 'flags': {}, 'order': number - 1, 'mode': 0,
                'inputs': [{'name': name, 'type': typ, 'link': None} for name, typ in (inputs or [])],
                'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in (outputs or [])],
                'properties': {'Node name for S&R': node_type}, 'widgets_values': widgets or []}
        self.nodes.append(item)
        return item

    def connect(self, origin, output, target, input_number):
        number = len(self.links) + 1
        typ = origin['outputs'][output]['type']
        assert typ == target['inputs'][input_number]['type']
        self.links.append([number, origin['id'], output, target['id'], input_number, typ])
        origin['outputs'][output]['links'].append(number)
        target['inputs'][input_number]['link'] = number

    def group(self, title, box, color='#3f789e'):
        self.groups.append({'title': title, 'bounding': box, 'color': color, 'font_size': 22,
                            'flags': {}})

    def save(self, name):
        obj = {'last_node_id': len(self.nodes), 'last_link_id': len(self.links), 'nodes': self.nodes,
               'links': self.links, 'groups': self.groups, 'config': {},
               'extra': {'ds': {'scale': 0.7, 'offset': [70, 100]}}, 'version': 0.4}
        dest = PACKAGE / 'workflows' / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    w = Workflow()
    voice=w.node('IndexTTSReference','音频参考 · 在此选择音色 / 音色＋情绪',[0,0],
        inputs=[('audio_in','AUDIO')],outputs=[('reference','INDEX_REFERENCE')],widgets=[VOICE_ONLY,''],size=[440,320])
    text=w.node('PrimitiveStringMultiline','生成正文 · 只填写要朗读的文字',[520,0],
        outputs=[('STRING','STRING')],widgets=['等一下……你听见了吗？刚才那个声音，明明是从房间里传出来的。别开灯，先把门锁上。握紧我的手，我们一起往后退。'],size=[440,260])
    params=w.node('IndexTTSParameters','情绪描述与推理参数',[520,340],
        inputs=[('emotion_vector','INDEX_EMOTION_VECTOR')],outputs=[('parameters','INDEX_PARAMETERS')],
        widgets=[v[1]['default'] for v in PARAMETERS.values()],size=[440,730])
    gen=w.node('IndexTTSGenerate','Index 2.0 / 2.5 · 单次克隆或默认音色生成',[1020,80],
        inputs=[('text','STRING'),('reference','INDEX_REFERENCE'),('parameters','INDEX_PARAMETERS')],
        outputs=[('audio','AUDIO')],widgets=[42,'fixed'],size=[450,180])
    w.connect(text,0,gen,0);w.connect(voice,0,gen,1);w.connect(params,0,gen,2)
    for row,kind in enumerate(('PreviewAudio','SaveAudioAdvanced')):
        out=w.node(kind,'试听' if row==0 else '保存无损 FLAC',[1550,row*280],
            inputs=[('audio','AUDIO')],outputs=[('audio','AUDIO')],widgets=[] if row==0 else ['audio/IndexTTS','flac'],size=[390,220])
        w.connect(gen,0,out,0)
    emotion=w.node('IndexTTSEmotionVector','八维情绪 · emotion_text 为空时生效',[0,460],
        outputs=[('emotion_vector','INDEX_EMOTION_VECTOR')],widgets=[0.0 for _ in EMOTIONS],size=[440,330])
    w.connect(emotion,0,params,0)
    w.group('一个音频参考与可选八维情绪',[-25,-60,490,920])
    w.group('正文与参数 · 描述优先于八维滑块',[495,-60,490,1200],'#686a78')
    w.group('服务端 GPU 推理 · ComfyUI CPU',[995,-60,980,680],'#407a53')
    w.save('01_IndexTTS_音色与情绪控制.json')

if __name__ == '__main__':
    main()
