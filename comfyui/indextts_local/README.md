# IndexTTS Local 0.2.1

一个音频参考节点，一份正文，一个参数节点。连接现有 `index_api.py`，共用 IndexTTS 2.0 / 2.5；ComfyUI 使用 CPU，模型在独立 GPU 服务推理。

## 安装或更新

将 `indextts_local` 文件夹放入 `ComfyUI/custom_nodes/`。在 ComfyUI 虚拟环境中执行：

```bash
python -m pip install -r custom_nodes/indextts_local/requirements.txt
```

更新后重启 ComfyUI、刷新浏览器，重新导入 `workflows/01_IndexTTS_音色与情绪控制.json`。本版重建了输入，不迁移旧工作流 widget 值；独立情绪音频参考已移除；八维滑块放入单独节点，连接参数节点。服务端 API 和启动脚本不需要修改。

## 使用

1. 音频参考：上传、下拉选择或连接标准 ComfyUI AUDIO，连线优先；音频需 0.1–15 秒，推荐 6–12 秒，不需要参考原文。
2. 在音频节点选择参考方式。
3. 正文只填写要朗读的内容；情绪描述放参数节点的 `emotion_text`。
4. 执行后试听或保存。

情绪来源固定按以下优先级决定，不需要额外的模式选择开关：

| 优先级 | 条件 | 情绪来源 | 被忽略的设置 |
|---|---|---|---|
| 1 | 音频节点选“参考音色＋情绪” | 同一份参考音频 | emotion_text 和八维滑块 |
| 2 | 仅参考音色，emotion_text 非空 | 参数节点的 emotion_text | 八维滑块 |
| 3 | 仅参考音色，emotion_text 为空 | 八维情绪节点 | emotion_text |

八维节点包含 `happy`、`angry`、`sad`、`afraid`、`disgusted`、`melancholic`、`surprised`、`calm`，分别为高兴、愤怒、悲伤、恐惧、反感、低落、惊讶、平静。输出接参数节点的 `emotion_vector`。滑块范围均为 0–1，顺序和归一化交给既有 API 处理。

新版工作流已连接八维节点，`emotion_text` 默认留空。填写描述后滑块置灰，清空描述则恢复；切到“参考音色＋情绪”时描述和滑块都失效，原值均保留。使用旧画布时若描述仍是“自然、平静。”，它作为非空描述会优先于滑块，需要清空才能使用滑块。

滑块全 0，或描述为空且没有连接八维节点时，按全 0 向量处理，不额外加强任何向量情绪。不会退回从正文分析情绪，也不会自动补写情绪描述。

“仅参考音色”使用 IndexTTS 的文字或向量情绪控制能力，不承诺模型能完全剥离参考音频中的表演特征。“参考音色＋情绪”按 1.0 跟随该音频，参数中的文字/滑块情绪强度也失效。

音频可留空，使用服务器默认参考；参考节点留空仍保留其参考方式。不连接参考节点时默认“仅参考音色”。服务器必须配置默认参考，这不是真正无参考生成。

## API 对应关系

- 仅参考音色且描述非空：`emotion_mode=text`，描述只来自参数节点。
- 仅参考音色且描述为空：`emotion_mode=vector`，发送八维节点的向量；未连接时为全 0。向量由 API 的官方归一化方法处理，不在客户端重复归一化。
- 参考音色＋情绪：相同字节同时发送为 `reference_audio` 和 `emotion_audio`，`emotion_mode=audio`；情绪描述清空。
- 音频留空且选择音色＋情绪：`emotion_mode=reference`，服务器将默认音频同时用于音色和情绪；描述清空。
- 正文仅传入 `text`，不用于自动情绪分析。

客户端连接我们部署的 JSON/WAV 适配器，地址由 `INDEX_TTS_API_URL` 设置，默认 `http://127.0.0.1:18081`。模型切换由后台脚本控制，工作流共用。

2.0 仅选 ZH/EN，时长系数保持 1；2.5 可选 ZH/EN/JA/ES/AR 并调整时长系数。一次执行发一次请求，没有多轮重生。

参数节点共用时，只有全部下游生成节点都选择音色＋情绪，描述才整体置灰；八维节点共用时，只要有一个下游仍需使用滑块，就允许编辑。每个生成节点独立遵守固定优先级。

## 验证

在已有 NumPy、SoundFile、PyTorch 的环境运行：

```bash
python -m unittest discover -s comfyui/indextts_local/tests -v
```

测试使用本机 HTTP 服务检查实际请求、音频输出、模式优先级和空参考，不加载 GPU 模型。
