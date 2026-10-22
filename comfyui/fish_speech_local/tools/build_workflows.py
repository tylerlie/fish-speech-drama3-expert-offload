"""Generate three importable ComfyUI 0.4 workflow templates."""
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from fish_speech_local.parameters import PARAMETER_INPUTS


def add_parameters(w, generator, pos, regeneration=False):
    generator['inputs'].append({'name': 'parameters', 'type': 'DRAMA_API_PARAMETERS', 'link': None})
    values = [options.get('default', kind[0] if isinstance(kind, list) else None)
              for kind, options in PARAMETER_INPUTS.values()]
    if regeneration:
        values[list(PARAMETER_INPUTS).index('single_chunk')] = True
    params = w.node('DramaAPIParameters', 'API 参数 · 三种生成节点通用', pos,
                    outputs=[('parameters', 'DRAMA_API_PARAMETERS')], widgets=values, size=[410, 520])
    w.connect(params, 0, generator, len(generator['inputs'])-1)
    w.group('API 参数 · 可共用', [pos[0]-20, pos[1]-45, 450, 600], '#686a78')
    return params


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
    for dialogue in (False, True):
        w = Workflow(); nodes=[]
        for i in range(9 if dialogue else 1):
            x,y=((i%3)*450,(i//3)*420) if dialogue else (0,0)
            nodes.append(w.node('DramaReference',f'Reference {i+1} · 留空即跳过' if dialogue else 'Reference · 留空即普通生成',
                         [x,y],inputs=[('audio_in','AUDIO')],outputs=[('reference','DRAMA_REFERENCE')],widgets=['',''],size=[390,355]))
        hx=1380 if dialogue else 460
        hub=w.node('DramaReferences','References · 空参考自动忽略',[hx,240 if dialogue else 50],
                   inputs=[(f'reference_{i}','DRAMA_REFERENCE') for i in range(1,10)],
                   outputs=[('references','DRAMA_REFERENCES')],size=[300,260])
        for i,n in enumerate(nodes):w.connect(n,0,hub,i)
        tx=hx+380
        text=w.node('PrimitiveStringMultiline','生成文稿 · 角色名：正文' if dialogue else '生成文稿 · 直接写正文',[tx,0],
                    outputs=[('STRING','STRING')],widgets=['小明：你准备好了吗？\n小红：[开心]准备好了，我们出发吧。' if dialogue else '你好，这是语音生成测试。参考音频和原文均为空时，直接生成语音。'],size=[420,280])
        gen=w.node('DramaMultiVoice' if dialogue else 'DramaSingleVoice','多角色 · 至少填写一个 Reference' if dialogue else '单音色 · 自动判断克隆或生成',[tx,380],
                   inputs=[('text','STRING'),('references','DRAMA_REFERENCES')],outputs=[('audio','AUDIO')],widgets=[42,'fixed'] if dialogue else [42,'fixed',4],size=[400,165] if dialogue else [400,195])
        w.connect(text,0,gen,0);w.connect(hub,0,gen,1)
        add_parameters(w, gen, [hx, 740] if dialogue else [0, 470], regeneration=not dialogue)
        for i,kind in enumerate(('PreviewAudio','SaveAudioAdvanced')):
            out=w.node(kind,'试听结果' if i==0 else '保存音频 · 无损 FLAC',[tx+500,i*235],inputs=[('audio','AUDIO')],outputs=[('audio','AUDIO')],
                       widgets=[] if i==0 else ['audio/DramaMulti' if dialogue else 'audio/DramaSingle','flac'],size=[360,180])
            w.connect(gen,0,out,0)
        w.group('九个参考槽 · 按需填写，无需改线' if dialogue else '参考可空 · 上传录音与填写原文',[-25,-60,1345 if dialogue else 440,1300 if dialogue else 450])
        w.group('参考汇总',[hx-20,170 if dialogue else -15,340,370]);w.group('生成与输出',[tx-25,-60,930,660],'#407a53')
        w.save('02_多角色长文本.json' if dialogue else '01_单音色克隆或生成.json')
    build_loop_workflow()


def build_loop_workflow():
    w = Workflow()
    references = []
    for i in range(9):
        title = f'Reference {i+1} · ' + ('小明' if i == 0 else '小红' if i == 1 else '可留空')
        references.append(w.node('DramaReference', title, [(i % 3)*450, (i // 3)*420], inputs=[('audio_in','AUDIO')],
            outputs=[('reference', 'DRAMA_REFERENCE')], widgets=['', ''], size=[390, 355]))
    hub = w.node('DramaReferences', 'References · 保留原九路汇总', [1380, 240],
        inputs=[(f'reference_{i}', 'DRAMA_REFERENCE') for i in range(1, 10)],
        outputs=[('references', 'DRAMA_REFERENCES')], size=[300, 260])
    for i, ref in enumerate(references):
        w.connect(ref, 0, hub, i)
    text = w.node('PrimitiveStringMultiline', '每行一段 · 角色名：[语气]台词', [1760, 0],
        outputs=[('STRING', 'STRING')], size=[430, 320], widgets=[
            '小明：你准备好了吗？接下来要测试语音了。\n'
            '小红：[开心]准备好了，我们出发吧。\n'
            '小明：[低声]路上小心一点。\n'
            '小红：[兴奋]好，马上出发！'])
    loop = w.node('DramaLoopReference', '循环 Reference · 按行单人克隆后拼接', [1760, 410],
        inputs=[('text', 'STRING'), ('references', 'DRAMA_REFERENCES')],
        outputs=[('完整拼接', 'AUDIO'), ('分段数组', 'AUDIO')],
        widgets=[42, 'fixed', 200, 4], size=[430, 240])
    # A standard AUDIO list lets native output nodes execute once per segment.
    loop['outputs'][1]['shape'] = 6
    w.connect(text, 0, loop, 0)
    w.connect(hub, 0, loop, 1)
    add_parameters(w, loop, [1380, 740], regeneration=True)
    for slot, x, label, prefix in [(0, 2280, '完整拼接', 'audio/DramaLoop/full'),
                                    (1, 2740, '分段数组 · 逐段', 'audio/DramaLoop/segments')]:
        for row, kind in enumerate(('PreviewAudio', 'SaveAudioAdvanced')):
            output = w.node(kind, label + ('试听' if row == 0 else '保存'), [x, row*280],
                inputs=[('audio', 'AUDIO')], outputs=[('audio', 'AUDIO')],
                widgets=[] if row == 0 else [prefix, 'flac'], size=[390, 220])
            w.connect(loop, slot, output, 0)
    w.group('参考音色 · 至少填一个，最多九个，其余留空', [-25, -60, 1345, 1300])
    w.group('原有汇总节点', [1360, 170, 340, 370])
    w.group('逐行生成 · 每个非空行都写角色名', [1735, -60, 495, 740], '#407a53')
    w.group('完整音频', [2255, -60, 440, 620], '#407a53')
    w.group('分段数组 · 不含额外段间停顿', [2715, -60, 440, 620], '#776398')
    w.save('03_拼接式多角色多语气.json')


if __name__=='__main__':main()
