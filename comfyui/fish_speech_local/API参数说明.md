# 自部署 Fish API 参数节点

版本：1.3.0；日期：2026-10-21。面向当前 `fish-speech` / `fish-speech-drama3-expert-offload` 的 `POST /v1/tts`，不增加托管服务专用字段，不修改 Fish 后端。

## 接线

```text
Fish · API 参数（自部署） · parameters ┬→ 单音色克隆 / 生成 · parameters
                                    ├→ 多角色长文本 · parameters
                                    └→ 循环 Reference · parameters
```

连接是可选的。同一个参数节点可同时连给三个生成节点。各节点复制配置后组装请求，不会修改共用配置。三个交付工作流均已接好参数节点；旧工作流不接参数时保持原请求行为。

## 参数对照

| 请求字段 | 本节点默认 | 说明 |
|---|---|---|
| `text` | 原文本节点 | 正文不放入参数节点，继续从原生 Text 输入 |
| `references` | 原 Reference/References | 音频、参考原文及角色映射沿用原节点 |
| `temperature` | 1.0 | 0.1–1.0；服务原始默认0.7；明确发送浮点值 |
| `top_p` | 0.9 | 0.1–1.0；服务原始默认0.7 |
| `max_new_tokens` | 1024 | 每个内部文本块的生成帧数上限；0使用模型内置上限，仍受上下文容量限制 |
| `seed` | 沿用生成节点 | 由 seed_mode 选择来源；不指定时请求省略 seed 字段 |
| `single_chunk` | false | 关闭单次请求的内部分块；不改变循环节点的逐行请求划分 |
| `use_memory_cache` | off | on/off；缓存参考音频编码，不缓存生成结果，不是前文上下文开关 |
| `reference_id` | 空 | 空时不发送；非空时使用自部署服务保存的参考ID，仅用于单音色且本地参考全空 |
| `format` | wav | wav/mp3/opus/pcm；HTTP响应编码，不是最终文件保存格式 |
| `streaming` | false | 仅wav可配true；接收完后输出完整AUDIO |
| `chunk_length` | 300 | 接口范围100–1000；当前Drama 3忽略该值，S2-Pro才使用 |
| `repetition_penalty` | 1.2 | 接口范围0.9–2.0；当前Drama 3/S2-Pro忽略，使用内部重复感知采样 |
| `normalize` | true | 接口接受；当前开源推理路径未执行数字日期规范化，也不是响度归一化 |
| `latency` | normal | normal/balanced；接口接受但当前开源推理路径未使用 |

`seed_mode` 是插件控件，不是Fish请求字段：

- **生成节点种子**：使用各生成节点原有 seed 控件，参数节点内 seed 数值不参与请求。
- **参数节点种子**：以参数节点内 seed 覆盖生成节点 seed。
- **不指定种子**：请求不包含 seed，不由插件设种子；不保证每次结果都不同。

循环节点只对种子的有效起点逐段加1，超过32位上限归零；空行不占种子。其余参数在每行请求中保持一致，原文角色、语气与参考按原规则处理。

## 音频格式

三个生成节点都输出标准 ComfyUI AUDIO。生成节点选择MP3不意味着保存节点一定输出MP3：API返回的MP3会先解码，保存节点再按自身选择写文件；默认WAV中转可避免多次有损编码。

PCM无文件头。本插件针对本项目Drama 3，按 **44100Hz、单声道、16位小端PCM** 解码。如果更换了输出采样率不同的后端，不应使用此PCM设置；WAV/MP3/Opus从文件头读取采样率。Opus后端编码会重采样到48kHz。

流式WAV先接收完整响应，再修正未知长度的标准RIFF头并解码。它不改变ComfyUI执行模型，不会让预览节点实时播放未完成音频。任何流式与非WAV的组合会在请求前报错。

## reference_id 的范围

ID是当前自部署Fish的已保存参考目录标识，不是公网Fish账号的音色ID。本节点中ID非空时，单音色工作流的Reference音频和原文都应留空。若同时连接有效参考，明确报错，避免服务端ID优先规则让用户误用另一份音色。

原生多角色和循环拼接均依赖Reference中的角色名与音频；一个全局ID无法表达这套角色绑定，因此明确拒绝非空ID。多角色时将参数节点的reference_id留空即可复用其余配置。

## 未加入的托管字段

当前开源请求没有 `sample_rate`、`mp3_bitrate`、`min_chunk_length`、`early_stop_threshold`、`condition_on_previous_chunks`、`prosody`、`features` 等字段，因此不创建对应假开关。`top_k` 内部固定30，接口未暴露，也不加入。Auto Tag是另外的文本标注功能，本次不包含。

单次请求内部的历史音频仍由Drama 3自动处理；循环节点跨行不继承生成音频。本版不改变模型、分段算法或采样实现。

## 验证实际请求

comfyui.log 中的“Fish API 请求参数”会打印实际发送的采样、格式、种子及兼容字段，省略正文、参考音频和已保存参考ID。服务默认值可用 `GET /v1/tts/defaults` 查询，但其中 `chunk_length`、重复惩罚等字段不代表Drama 3会使用。
