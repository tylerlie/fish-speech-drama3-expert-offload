import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

let extension;
const source = fs.readFileSync(new URL('../web/parameters.js', import.meta.url), 'utf8');
vm.runInNewContext(source.replace(/^import .*;\s*$/gm, ''), {
  app: { registerExtension(value) { extension = value; } },
});
for (const name of ['DramaSingleVoice', 'DramaMultiVoice', 'DramaLoopReference']) {
  class Generator {
    constructor() { this.inputs = []; this.configured = 0; this.widgets = name==='DramaMultiVoice' ? [] : [{name:'regenerate_rounds',value:4}]; }
    addInput(name,type) { const port = { name,type,link:null }; this.inputs.push(port); return port; }
    onConfigure() { this.configured++; }
  }
  await extension.beforeRegisterNodeDef(Generator, { name });
  const node = new Generator(); node.onNodeCreated();
  if (name!=='DramaMultiVoice') {
    assert.equal(node.widgets[0].label, '重生轮次 · 双参考');
    assert.equal(node.widgets[0].name, 'regenerate_rounds');
    assert.equal(node.widgets[0].value, 4);
  }
  assert.equal(node.inputs.length, 1);
  assert.equal(node.inputs[0].name, 'parameters');
  assert.equal(node.inputs[0].type, 'DRAMA_API_PARAMETERS');
  node.inputs = []; node.onConfigure();
  assert.equal(node.inputs.length, 1);
  assert.equal(node.configured, 1);
  node.inputs[0].link = 123; node.onConfigure();
  assert.equal(node.inputs.length, 1);
  assert.equal(node.inputs[0].link, 123);
}
class Parameters {
  constructor() { this.widgets = [{name:'temperature'}, {name:'normalize'}, {name:'seed_mode'}]; this.size=[300,300]; }
  setSize(size) { this.size = size; }
}
await extension.beforeRegisterNodeDef(Parameters, { name:'DramaAPIParameters' });
const node = new Parameters(); node.onNodeCreated();
assert.match(node.widgets[0].label, /Temperature/);
assert.match(node.widgets[1].label, /未实现/);
assert.match(node.widgets[2].label, /种子来源/);
console.log('API parameters UI: three optional ports, old templates, existing links and labels passed.');
