import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { syncEmotionControls } from './parameters.js';

app.registerExtension({
  name: 'IndexTTSLocal.Reference',
  async beforeRegisterNodeDef(nodeType, definition) {
    if (definition.name !== 'IndexTTSReference') return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      const node = this;
      const ensureAudioInput = () => {
        let input = node.inputs?.find(input => input.name === 'audio_in');
        if (!input) input = node.addInput('audio_in', 'AUDIO');
        input.label = '外部音频（优先）';
      };
      ensureAudioInput();
      const audio = node.widgets.find(w => w.name === 'audio');
      const mode = node.widgets.find(w => w.name === 'reference_mode');
      if (!audio || !mode) return;
      audio.label = '参考音频'; mode.label = '参考方式';
      const root = document.createElement('div');
      root.dataset.indexReference = 'true';
      root.style.cssText = 'display:flex;flex-direction:column;gap:8px;padding:8px;box-sizing:border-box;background:#282a2d;border-radius:6px;width:100%;height:100%;color:#ddd;font:13px sans-serif';
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap';
      const upload = document.createElement('button');
      const clear = document.createElement('button');
      const reload = document.createElement('button');
      upload.textContent = '上传参考音频'; clear.textContent = '清空音频';
      reload.textContent = '刷新媒体列表';
      for (const b of [upload, clear, reload]) {
        b.type = 'button';
        b.style.cssText = 'padding:7px 8px;border:1px solid #666;border-radius:5px;background:#3b4047;color:white;cursor:pointer';
        row.append(b);
      }
      const picker = document.createElement('input');
      picker.type = 'file'; picker.accept = 'audio/*,video/*,.wav,.flac,.mp3,.m4a,.ogg,.aac,.opus,.aif,.aiff,.wma,.wv,.mp4,.webm,.mov,.mkv,.avi'; picker.hidden = true;
      const player = document.createElement('audio');
      player.controls = true; player.preload = 'metadata'; player.style.width = '100%';
      player.setAttribute('aria-label', '参考音频试听');
      const status = document.createElement('div');
      status.setAttribute('role', 'status');
      status.style.cssText = 'font-size:12px;overflow-wrap:anywhere;line-height:1.4';
      root.append(row, picker, player, status);
      let missingSelection = false;
      const updateChoices = (choices) => {
        const selected = String(audio.value || '');
        const files = choices.filter(value => typeof value === 'string' && value !== '');
        missingSelection = Boolean(selected && !files.includes(selected));
        if (missingSelection) files.push(selected); // Preserve old workflows; validation handles missing files.
        audio.options ||= {};
        audio.options.values = ['', ...new Set(files)];
      };
      const refresh = () => {
        const connected = node.inputs?.some(input => input.name === 'audio_in' && input.link != null);
        const filename = String(audio.value || '').trim();
        const followsEmotion = mode.value === '参考音色＋情绪';
        const purpose = followsEmotion
          ? '同一音频用于音色和情绪；参数中的情绪描述、强度不生效。'
          : '音色使用参考音频；情绪先取参数 emotion_text，描述为空才使用八维滑块。';
        if (connected) {
          player.pause(); player.removeAttribute('src'); player.load(); player.hidden = true;
          status.textContent = '当前使用上游音频，本地选择忽略。' + purpose;
        } else if (filename) {
          const parts = filename.replaceAll('\\', '/').split('/');
          const query = new URLSearchParams({filename: parts.pop(), subfolder: parts.join('/'), type: 'input'});
          const url = api.apiURL('/view?' + query);
          if (player.getAttribute('src') !== url) { player.src = url; player.load(); }
          player.hidden = false;
          status.textContent = missingSelection ? '列表中未找到当前文件，请重新选择或上传；原选择已保留。' : '当前使用所选音频（需 0.1–15 秒）。' + purpose;
        } else {
          player.pause(); player.removeAttribute('src'); player.load(); player.hidden = true;
          status.textContent = '音频留空，使用服务器默认参考。' + purpose;
        }
        syncEmotionControls(node.graph);
        node.setDirtyCanvas(true, true);
      };
      const previous = audio.callback;
      audio.callback = function () { previous?.apply(this, arguments); missingSelection = false; refresh(); };
      const modeChanged = mode.callback;
      mode.callback = function () { modeChanged?.apply(this, arguments); refresh(); };
      const reloadChoices = async () => {
        reload.disabled = true;
        try {
          const response = await api.fetchApi('/object_info/IndexTTSReference');
          if (!response.ok) throw new Error('HTTP ' + response.status);
          const result = await response.json();
          const choices = result.IndexTTSReference?.input?.optional?.audio?.[0];
          if (!Array.isArray(choices)) throw new Error('服务未返回媒体列表，请确认已升级插件并重启 ComfyUI');
          updateChoices(choices); refresh();
        } catch (error) {
          status.textContent = '刷新列表失败：' + error.message + '。原选择保留。';
        } finally { reload.disabled = false; }
      };
      reload.addEventListener('click', reloadChoices);
      upload.addEventListener('click', () => picker.click());
      picker.addEventListener('change', async () => {
        const file = picker.files?.[0]; if (!file) return;
        upload.disabled = clear.disabled = true; status.textContent = '正在上传……';
        try {
          const form = new FormData();
          form.append('image', file); form.append('type', 'input'); form.append('subfolder', 'index_reference');
          const response = await api.fetchApi('/upload/image', {method: 'POST', body: form});
          if (!response.ok) throw new Error('HTTP ' + response.status);
          const result = await response.json();
          audio.value = result.subfolder ? result.subfolder + '/' + result.name : result.name;
          updateChoices([...(audio.options?.values || []), audio.value]);
          refresh();
        } catch (error) {
          status.textContent = '上传失败：' + error.message + '。原参考保留，可重新上传。';
        } finally { picker.value = ''; upload.disabled = clear.disabled = false; }
      });
      clear.addEventListener('click', () => {
        audio.value = '';
        missingSelection = false;
        refresh();
      });
      node.addDOMWidget('reference_controls', 'INDEX_REFERENCE_CONTROLS', root,
        {serialize: false, hideOnZoom: false, getMinHeight: () => 120, getMaxHeight: () => 140});
      const configured = node.onConfigure;
      node.onConfigure = function () {
        configured?.apply(this, arguments);
        ensureAudioInput();  // Old templates do not yet serialize this optional port.
        updateChoices(audio.options?.values || []);
        queueMicrotask(refresh);
      };
      const connectionsChanged = node.onConnectionsChange;
      node.onConnectionsChange = function () { connectionsChanged?.apply(this, arguments); queueMicrotask(refresh); };
      const removed = node.onRemoved;
      node.onRemoved = function () { player.pause(); removed?.apply(this, arguments); };
      node.setSize([Math.max(node.size[0], 440), Math.max(node.size[1], 300)]);
      queueMicrotask(refresh);
    };
  }
});
