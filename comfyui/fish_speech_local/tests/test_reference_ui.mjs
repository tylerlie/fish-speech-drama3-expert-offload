// Exercise source selection and old-template port restoration without a browser.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

class Element {
  constructor() { this.style = {}; this.dataset = {}; this.children = []; this.handlers = {}; this.attrs = {}; }
  append(...children) { this.children.push(...children); }
  setAttribute(key, value) { this.attrs[key] = value; }
  getAttribute(key) { return this.attrs[key]; }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(name, fn) { this.handlers[name] = fn; }
  load() {}
  pause() {}
  set src(value) { this.attrs.src = value; }
}
let extension;
const pending = [];
let files = ['', 'voice.wav', 'fish_reference/已有录音.wav'];
let requestFails = false;
const requests = [];
const fakeAPI = {
  apiURL: path => path,
  async fetchApi(path, options) {
    requests.push(path);
    if (requestFails) return {ok:false,status:500};
    if (path === '/upload/image') return {ok:true,json:async()=>({subfolder:'fish_reference',name:'new.wav'})};
    assert.equal(path, '/object_info/DramaReference');
    return {ok:true,json:async()=>({DramaReference:{input:{optional:{audio:[files,{default:''}]}}}})};
  },
};
const source = fs.readFileSync(new URL('../web/reference.js', import.meta.url), 'utf8');
vm.runInNewContext(source.replace(/^import .*;\s*$/gm, ''), {
  app: { registerExtension(value) { extension = value; } },
  api: fakeAPI,
  document: { createElement: () => new Element() },
  URLSearchParams, FormData: class { append() {} }, queueMicrotask: fn => pending.push(fn),
});
const flush = () => { while (pending.length) pending.shift()(); };
class ReferenceNode {
  constructor() {
    this.inputs = [{ name: 'audio_in', type: 'AUDIO', link: null }];
    this.widgets = [{ name: 'audio', type:'combo', value: 'voice.wav', options:{values:[...files]} }, { name: 'text', value: '原文。', inputEl: {} }];
    this.size = [390, 355];
  }
  addInput(name, type) { const input = { name, type, link: null }; this.inputs.push(input); return input; }
  addDOMWidget(name, type, root) { this.controls = root; }
  setDirtyCanvas() {}
  setSize(value) { this.size = value; }
}
await extension.beforeRegisterNodeDef(ReferenceNode, { name: 'DramaReference' });
const node = new ReferenceNode();
node.onNodeCreated(); flush();
const [row, picker, player, status] = node.controls.children;
assert.match(status.textContent, /当前使用本地媒体/);
assert.equal(player.hidden, false);
assert.equal(node.inputs.length, 1);

node.inputs[0].link = 0;  // Link ID zero is connected too.
node.onConnectionsChange(); flush();
assert.match(status.textContent, /上游 audio_in/);
assert.equal(player.hidden, true);
assert.equal(player.getAttribute('src'), undefined);
assert.equal(node.widgets[0].value, 'voice.wav');

node.inputs[0].link = null;
node.onConnectionsChange(); flush();
assert.match(status.textContent, /当前使用本地媒体/);
assert.equal(player.hidden, false);

// Import a pre-1.2.0 serialized node which had no audio input ports.
node.inputs = [];
node.onConfigure(); flush();
assert.equal(node.inputs.length, 1);
assert.equal(node.inputs[0].name, 'audio_in');
assert.equal(node.inputs[0].type, 'AUDIO');
node.onConfigure(); flush();
assert.equal(node.inputs.length, 1);

node.inputs[0].link = 12;
row.children[1].handlers.click(); flush();
assert.equal(node.inputs[0].link, 12);
assert.equal(node.widgets[0].value, '');
assert.equal(node.widgets[1].value, '');
assert.match(status.textContent, /上游 audio_in/);

// Selecting an existing media item changes playback without uploading anything.
node.inputs[0].link = null;
node.widgets[0].value = 'fish_reference/已有录音.wav';
node.widgets[0].callback();
assert.match(player.getAttribute('src'), /subfolder=fish_reference/);
assert.equal(requests.length, 0);

files = ['', 'voice.wav', 'fish_reference/已有录音.wav', 'added.mp3'];
await row.children[2].handlers.click();
assert.ok(node.widgets[0].options.values.includes('added.mp3'));
assert.equal(node.widgets[0].value, 'fish_reference/已有录音.wav');
assert.equal(node.widgets[1].value, '');

// Removed files and failed refreshes must not silently choose another voice.
files = ['', 'voice.wav'];
await row.children[2].handlers.click();
assert.equal(node.widgets[0].value, 'fish_reference/已有录音.wav');
assert.ok(node.widgets[0].options.values.includes('fish_reference/已有录音.wav'));
assert.match(status.textContent, /未找到当前文件/);
requestFails = true;
await row.children[2].handlers.click();
assert.match(status.textContent, /刷新列表失败/);
assert.equal(node.widgets[0].value, 'fish_reference/已有录音.wav');
requestFails = false;

picker.files = [{}];
await picker.handlers.change();
assert.equal(node.widgets[0].value, 'fish_reference/new.wav');
assert.ok(node.widgets[0].options.values.includes('fish_reference/new.wav'));
assert.equal(requests.filter(path => path==='/upload/image').length, 1);
assert.equal(row.children[2].disabled, false);
console.log('Reference UI: media reuse, refresh, upload, selection preservation, upstream priority and old templates passed.');
