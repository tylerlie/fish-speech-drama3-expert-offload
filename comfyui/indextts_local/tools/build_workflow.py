"""Build the single IndexTTS 2.0/2.5 workflow template."""
import json
import sys
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from indextts_local.nodes import PARAMETERS

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
    voice=w.node('IndexTTSReference','音色参考 · 留空使用服务器默认音色',[0,0],
        inputs=[('audio_in','AUDIO')],outputs=[('reference','INDEX_REFERENCE')],widgets=['',''],size=[400,355])
    emotion=w.node('IndexTTSReference','情绪参考 · 仅 audio 情绪模式需要',[0,450],
        inputs=[('audio_in','AUDIO')],outputs=[('reference','INDEX_REFERENCE')],widgets=['',''],size=[400,355])
    text=w.node('PrimitiveStringMultiline','生成正文 · 不写 Fish 情绪标签',[490,0],
        outputs=[('STRING','STRING')],widgets=['你终于回来了！我还以为，再也见不到你了。回来就好，回来就好。'],size=[440,260])
    params=w.node('IndexTTSParameters','情绪与推理参数 · 两版共用',[490,340],
        outputs=[('parameters','INDEX_PARAMETERS')],widgets=[v[1]['default'] for v in PARAMETERS.values()],size=[440,850])
    gen=w.node('IndexTTSGenerate','Index 2.0 / 2.5 · 单次克隆或默认音色生成',[1020,80],
        inputs=[('text','STRING'),('reference','INDEX_REFERENCE'),('emotion_reference','INDEX_REFERENCE'),('parameters','INDEX_PARAMETERS')],
        outputs=[('audio','AUDIO')],widgets=[42,'fixed'],size=[450,180])
    w.connect(text,0,gen,0);w.connect(voice,0,gen,1);w.connect(emotion,0,gen,2);w.connect(params,0,gen,3)
    for row,kind in enumerate(('PreviewAudio','SaveAudioAdvanced')):
        out=w.node(kind,'试听' if row==0 else '保存无损 FLAC',[1550,row*280],
            inputs=[('audio','AUDIO')],outputs=[('audio','AUDIO')],widgets=[] if row==0 else ['audio/IndexTTS','flac'],size=[390,220])
        w.connect(gen,0,out,0)
    w.group('音频输入 · 上传 / 下拉复用 / 上游连线',[-25,-60,460,900])
    w.group('生成文稿与参数',[465,-60,490,1300],'#686a78')
    w.group('服务端 GPU 推理 · ComfyUI CPU',[995,-60,980,680],'#407a53')
    w.save('01_IndexTTS_克隆与默认音色生成.json')

if __name__ == '__main__':
    main()
