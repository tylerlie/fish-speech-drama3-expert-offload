import { app } from '../../scripts/app.js';

const labels = {
  emotion_text: '情绪描述 · 优先于滑块', emo_alpha: '文字 / 滑块情绪强度',
  happy: '高兴 · happy', angry: '愤怒 · angry', sad: '悲伤 · sad', afraid: '恐惧 · afraid',
  disgusted: '反感 · disgusted', melancholic: '低落 · melancholic',
  surprised: '惊讶 · surprised', calm: '平静 · calm',
  temperature: 'Temperature', top_p: 'Top P', top_k: 'Top K',
  num_beams: '搜索束数', repetition_penalty: '重复惩罚', length_penalty: '长度惩罚',
  max_mel_tokens: '每段音频 Token 上限', max_text_tokens_per_segment: '文本分段 Token 上限',
  interval_silence: '段间静音（毫秒）', use_random: '情绪样本随机选择',
  use_memory_cache: '参考特征缓存', lang: '语言', duration_factor: '时长系数（仅 2.5）',
};

function linkedNode(graph, node, inputName) {
  const port = node.inputs?.find(input => input.name === inputName);
  const link = port?.link != null ? graph.links?.[port.link] : null;
  return link ? graph.getNodeById(link.origin_id) : null;
}

export function syncEmotionControls(graph = app.graph) {
  if (!graph) return;
  const nodes = graph._nodes || [];
  const vectorNeeded = new Map();
  for (const parameters of nodes.filter(node => node.type === 'IndexTTSParameters')) {
    const consumers = nodes.filter(node => node.type === 'IndexTTSGenerate' &&
      linkedNode(graph, node, 'parameters') === parameters);
    const following = consumers.filter(node => {
      const reference = linkedNode(graph, node, 'reference');
      return reference?.widgets?.find(widget => widget.name === 'reference_mode')?.value === '参考音色＋情绪';
    }).length;
    // A shared parameters node must stay editable if any consumer needs text.
    const disabled = consumers.length > 0 && following === consumers.length;
    const hasDescription = Boolean(String(parameters.widgets?.find(w => w.name === 'emotion_text')?.value || '').trim());
    vectorNeeded.set(parameters, !disabled && !hasDescription);
    for (const widget of parameters.widgets || []) {
      if (!['emotion_text', 'emo_alpha'].includes(widget.name)) continue;
      widget.disabled = disabled;
      if (widget.inputEl) {
        widget.inputEl.disabled = disabled;
        widget.inputEl.style.opacity = disabled ? '0.5' : '';
      }
    }
    if (parameters.indexEmotionStatus) {
      parameters.indexEmotionStatus.textContent = disabled
        ? '参考音频提供音色＋情绪：情绪描述与八维滑块均不参与推理。'
        : following > 0
          ? '参数被多个生成节点共用：仅参考音色的节点先用描述，描述为空才用滑块。'
          : hasDescription
            ? '当前使用情绪描述；八维滑块不参与推理，数值保留。正文只用于朗读。'
            : '情绪描述为空：使用八维滑块。未连接时按八维全 0 处理，正文不用于分析情绪。';
    }
    parameters.setDirtyCanvas?.(true, true);
  }
  for (const vector of nodes.filter(node => node.type === 'IndexTTSEmotionVector')) {
    const consumers = [...vectorNeeded.keys()].filter(parameters => linkedNode(graph, parameters, 'emotion_vector') === vector);
    const disabled = consumers.length > 0 && consumers.every(parameters => !vectorNeeded.get(parameters));
    for (const widget of vector.widgets || []) {
      if (!['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm'].includes(widget.name)) continue;
      widget.disabled = disabled;
      if (widget.inputEl) widget.inputEl.disabled = disabled;
    }
    if (vector.indexEmotionStatus) vector.indexEmotionStatus.textContent = disabled
      ? '当前被参考音频情绪或情绪描述覆盖；滑块数值保留。'
      : '仅参考音色且 emotion_text 为空时生效。全 0 表示不额外加强任何向量情绪。';
    vector.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: 'IndexTTSLocal.Parameters',
  async beforeRegisterNodeDef(nodeType, definition) {
    if (!['IndexTTSParameters', 'IndexTTSGenerate', 'IndexTTSEmotionVector'].includes(definition.name)) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const ensureVectorInput = () => {
        if (definition.name !== 'IndexTTSParameters') return;
        let port = this.inputs?.find(input => input.name === 'emotion_vector');
        if (!port) port = this.addInput('emotion_vector', 'INDEX_EMOTION_VECTOR');
        port.label = '八维情绪（描述为空时生效）';
      };
      ensureVectorInput();
      if (['IndexTTSParameters', 'IndexTTSEmotionVector'].includes(definition.name)) {
        for (const widget of this.widgets || []) {
          if (labels[widget.name]) widget.label = labels[widget.name];
        }
        const status = document.createElement('div');
        status.style.cssText = 'padding:8px;color:#ddd;font:13px/1.5 sans-serif;white-space:normal';
        status.setAttribute('role', 'status');
        this.indexEmotionStatus = status;
        this.addDOMWidget('emotion_status', 'INDEX_EMOTION_STATUS', status,
          {serialize: false, hideOnZoom: false, getMinHeight: () => 58, getMaxHeight: () => 72});
        const description = this.widgets?.find(widget => widget.name === 'emotion_text');
        if (description) {
          const callback = description.callback;
          description.callback = (...args) => {
            callback?.apply(description, args);
            syncEmotionControls(this.graph);
          };
          description.inputEl?.addEventListener('input', () => queueMicrotask(() => syncEmotionControls(this.graph)));
        }
      }
      const configured = this.onConfigure;
      this.onConfigure = function () {
        configured?.apply(this, arguments);
        ensureVectorInput();
        queueMicrotask(() => syncEmotionControls(this.graph));
      };
      const changed = this.onConnectionsChange;
      this.onConnectionsChange = function () {
        changed?.apply(this, arguments);
        queueMicrotask(() => syncEmotionControls(this.graph));
      };
      const removed = this.onRemoved;
      this.onRemoved = function () {
        removed?.apply(this, arguments);
        queueMicrotask(() => syncEmotionControls());
      };
      queueMicrotask(() => syncEmotionControls(this.graph));
    };
  },
});
