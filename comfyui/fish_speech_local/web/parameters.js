import { app } from '../../scripts/app.js';

const labels = {
  temperature: '表现变化 · Temperature', top_p: '多样性 · Top P',
  max_new_tokens: '每段生成帧数上限', seed_mode: '种子来源', seed: '参数节点种子',
  single_chunk: '整次请求不分块', use_memory_cache: '参考编码缓存', reference_id: '已保存音色 ID（可空）',
  format: 'API 音频格式', streaming: 'API 流式请求（仅 WAV）',
  chunk_length: '分段长度（Drama 3 忽略）', repetition_penalty: '重复惩罚（当前忽略）',
  normalize: '文本规范化（当前未实现）', latency: '延迟模式（当前未实现）',
};

app.registerExtension({
  name: 'FishSpeechLocal.APIParameters',
  async beforeRegisterNodeDef(nodeType, definition) {
    const generator = ['DramaSingleVoice', 'DramaMultiVoice', 'DramaLoopReference'].includes(definition.name);
    if (!generator && definition.name !== 'DramaAPIParameters') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      if (generator) {
        const ensurePort = () => {
          let port = this.inputs?.find(input => input.name === 'parameters');
          if (!port) port = this.addInput('parameters', 'DRAMA_API_PARAMETERS');
          port.label = 'API 参数（可选）';
          const rounds = this.widgets?.find(widget => widget.name === 'regenerate_rounds');
          if (rounds) rounds.label = '重生轮次 · 双参考';
        };
        ensurePort();
        const configured = this.onConfigure;
        this.onConfigure = function () { configured?.apply(this, arguments); ensurePort(); };
      } else {
        for (const widget of this.widgets || []) {
          if (labels[widget.name]) widget.label = labels[widget.name];
        }
        this.setSize([Math.max(this.size[0], 410), Math.max(this.size[1], 500)]);
      }
    };
  },
});
