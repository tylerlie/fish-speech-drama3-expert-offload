const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/parameters.js'), 'utf8')
  .replace(/^import .*;\n/gm, '').replace('export function ', 'function ');
const app = { registerExtension() {} };
const context = vm.createContext({ app, queueMicrotask });
vm.runInContext(source, context);
const parameters = { id: 1, type: 'IndexTTSParameters', widgets: [
  { name: 'emotion_text', value: '保留这段描述', inputEl: { style: {} } },
  { name: 'emo_alpha', value: .25 },
], indexEmotionStatus: {}, setDirtyCanvas() {} };
parameters.inputs = [{ name: 'emotion_vector', link: 3 }];
const vector = { id: 5, type: 'IndexTTSEmotionVector', widgets: [{ name: 'angry', value: .8 }], indexEmotionStatus: {}, setDirtyCanvas() {} };
const reference = { id: 2, type: 'IndexTTSReference', widgets: [{ name: 'reference_mode', value: '参考音色＋情绪' }] };
const generator = { id: 3, type: 'IndexTTSGenerate', inputs: [{ name: 'parameters', link: 1 }, { name: 'reference', link: 2 }] };
const graph = { _nodes: [parameters, reference, generator, vector], links: {1: {origin_id: 1}, 2: {origin_id: 2}, 3: {origin_id: 5}}, getNodeById(id) { return this._nodes.find(n => n.id === id); } };
app.graph = graph;
context.syncEmotionControls(graph);
assert.equal(parameters.widgets[0].inputEl.disabled, true);
assert.equal(parameters.widgets[1].disabled, true);
assert.equal(parameters.widgets[0].value, '保留这段描述');
assert.equal(vector.widgets[0].disabled, true);
reference.widgets[0].value = '仅参考音色';
context.syncEmotionControls(graph);
assert.equal(parameters.widgets[0].inputEl.disabled, false);
assert.equal(parameters.widgets[0].value, '保留这段描述');
assert.equal(vector.widgets[0].disabled, true, 'nonempty description overrides sliders');
parameters.widgets[0].value = ' \n';
context.syncEmotionControls(graph);
assert.equal(vector.widgets[0].disabled, false, 'blank description enables sliders');
assert.equal(vector.widgets[0].value, .8);
parameters.widgets[0].value = '保留这段描述';
reference.widgets[0].value = '参考音色＋情绪';
graph._nodes.push({ id: 4, type: 'IndexTTSGenerate', inputs: [{ name: 'parameters', link: 1 }] });
context.syncEmotionControls(graph);
assert.equal(parameters.widgets[0].inputEl.disabled, false, 'shared parameters must remain editable for a text consumer');
assert.match(parameters.indexEmotionStatus.textContent, /多个/);
graph._nodes.pop();
graph.links[2] = null;
context.syncEmotionControls(graph);
assert.equal(parameters.widgets[0].inputEl.disabled, false);
console.log('PASS: mode switching, value preservation, shared parameters, disconnected reference');
