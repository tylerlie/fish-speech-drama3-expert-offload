# Fish Speech · Drama 3 Expert Offload

基于 [Fish Speech 官方仓库](https://github.com/fishaudio/fish-speech)的社区改造项目，目标是在 **单张 RTX 5090 32GB** 上运行 Drama 3，保留官方 Fish API，并继续使用现有 ComfyUI 插件和工作流。

仓库现已包含 **Fish Speech Local 1.4.0** ComfyUI 插件，位于 [`comfyui/fish_speech_local`](comfyui/fish_speech_local/)。支持单音色克隆、普通生成、逐行多角色拼接及双参考重生，安装和用法见下文。

改造方向是将路由专家的原始 FP8 权重保存在 CPU 内存中，使用固定预算的 GPU 专家缓存按需调入并执行。其余模型部分优先常驻 GPU，不额外进行 INT4 量化，不裁剪专家，也不减少原模型每次选择的专家数量。

> **当前状态：已包含同步专家卸载实现与 ComfyUI 插件。** 代码接入不等于完成单张 5090 32GB 的完整验收；速度、音质与显存峰值仍需在目标 GPU、模型和实际工作负载下验证。

## 项目目标

- 使用原 Drama 3 检查点，保留已有 FP8 专家权重及缩放参数。
- 通过 CPU 内存与 GPU 缓存控制专家权重的显存占用。
- 保持官方路由、共享专家、提示构造、采样和音频编解码逻辑。
- 保持 `/v1/health`、`/v1/tts` 及现有请求、响应格式兼容，兼容性需在改造后回归验证。
- ComfyUI 继续以 CPU 模式运行，现有插件、节点和工作流无需因卸载方案而修改。
- 保留原版执行路径，通过显式配置启用卸载，便于对照测试和回退。

Drama 3 本身已经是 MoE 模型。本项目改造的是专家的存储和执行调度，不改变模型架构，也不训练新模型。

## 部署与工作流

目标部署为一个云容器、两个服务：

```text
ComfyUI（CPU）
    │ 现有插件与工作流
    ▼
Fish Speech API（0.0.0.0:18080）
    │ 官方推理流程 + 专家卸载
    ├── GPU：非专家部分、音频模块、专家缓存
    └── CPU 内存：完整路由专家权重
```

沿用现有部署布局：

```text
/root/
├── fish-speech/       本项目与 .fish 虚拟环境、模型
├── ComfyUI/           ComfyUI 与 .comfy 虚拟环境、插件
├── run.sh            先启动并预热 Fish，再启动 ComfyUI
├── fish.log
└── comfyui.log
```

API 监听地址和端口由启动配置指定。服务只有在模型加载及真实推理预热成功后，才应被启动脚本判定为就绪。

仓库内提供以下三类工作流，插件调用同一 Fish 语音后端：

| 工作流 | 输入与行为 | 改造后的验收重点 |
| --- | --- | --- |
| 单音色克隆 / 生成 | 参考音频与对应文本可空；为空时生成语音 | 有参考与无参考两条路径 |
| 原生多角色长文本 | 沿用现有角色集合与文本格式 | 接口回归；现有漏读问题单独记录 |
| 拼接式多角色多语气 | 按非空换行逐次发送单人克隆请求，最多 9 个克隆角色 | 角色切换、语气、完整拼接与音频列表 |

逐行循环可以复用同一份模型与专家缓存，无需为每个角色加载一份模型。请求的参考信息、生成状态和随机种子必须保持隔离。专家卸载本身不意味着原生多角色漏读已修复。

## ComfyUI 插件安装与使用

### 安装

插件源码、原生节点工作流和测试均在 [`comfyui/fish_speech_local/`](comfyui/fish_speech_local/)。以下假设本仓库位于 `/root/fish-speech-drama3-expert-offload`，ComfyUI 位于 `/root/ComfyUI`；实际目录不同时替换对应路径。

先停止 ComfyUI，在其虚拟环境中复制插件并安装依赖：

```bash
source /root/ComfyUI/.comfy/bin/activate
mkdir -p /root/ComfyUI/custom_nodes
cp -a /root/fish-speech-drama3-expert-offload/comfyui/fish_speech_local /root/ComfyUI/custom_nodes/
python -m pip install -r /root/ComfyUI/custom_nodes/fish_speech_local/requirements.txt
```

正确位置为 `/root/ComfyUI/custom_nodes/fish_speech_local/__init__.py`。升级时覆盖同名目录，不要安装第二份插件。以后更新本仓库后，需再次复制插件目录；已经运行的 ComfyUI 不会自动重新加载 Python 代码。

插件默认连接 `http://127.0.0.1:18080`，可直接沿用已有的 `run.sh`。启动脚本应先等待 Fish 加载和预热完成，再启动 CPU ComfyUI。若手动启动 ComfyUI：

```bash
cd /root/ComfyUI
source .comfy/bin/activate
FISH_API_URL=http://127.0.0.1:18080 CUDA_VISIBLE_DEVICES="" \
  python main.py --cpu --listen 0.0.0.0 --port 8188 --disable-partner-nodes
```

`FISH_API_URL` 填服务根地址，不附加 `/v1/tts`；服务启用密钥时通过 `FISH_API_KEY` 配置。重启 ComfyUI 后刷新浏览器，导入下面的 JSON，或在模板中选择 `fish_speech_local` 下的工作流。插件不会在 ComfyUI 中加载 Fish 模型。

### 工作流与接线

| 工作流 | 用法 |
| --- | --- |
| [01：单音色克隆或生成](comfyui/fish_speech_local/workflows/01_单音色克隆或生成.json) | 参考音频与原文都空时普通生成；填写后单音色克隆。原文和生成文稿无需角色名前缀。 |
| [02：原生多角色长文本](comfyui/fish_speech_local/workflows/02_多角色长文本.json) | 至少绑定一个角色，最多九个；整篇交给 Fish。保留原请求方式，不支持插件双参考重生。 |
| [03：拼接式多角色多语气](comfyui/fish_speech_local/workflows/03_拼接式多角色多语气.json) | 每个非空物理行独立克隆，再拼接。输出完整 AUDIO 和按原文顺序排列的最终分段数组。 |

主要接线为：

```text
Reference（音频与对应原文）→ References（最多九路）→ 单音色 / 原生多角色 / 循环 Reference
原生 Text (Multiline) ─────────────────────────→ 生成节点的 text
API 参数节点 ──────────────────────────────────→ 生成节点的 parameters（可选）
生成节点的 AUDIO ──────────────────────────────→ 原生试听 / 保存音频
```

Reference 可以上传音频、下拉复用 ComfyUI `input/` 内已有媒体，也可以连接原生 Load Audio 或第三方 Load Audio by URL 的 AUDIO 输出。**连线后以上游音频为准**，本地选择不参与；原文必须对应实际使用的录音。未连线且音频、原文同时为空时，该槽自动跳过，无需填满九个节点。

多角色参考原文写作 `小明：这段录音实际说的话。`。工作流 03 的生成文稿示例：

```text
小明：[惊喜，声音发颤]你终于回来了！
小红：[轻声，释然]回来就好，回来就好。
小明：[压低声音，紧张]别出声，他还在门外。
```

每个非空行都需写角色名；换行决定分段，文本框视觉折行不算新行。参考中的同名样本归为同一角色，最多九份原始参考。循环节点的“分段数组”每行仅一项，不包含中间重生轮次；可直接连接原生试听或保存节点。

### 双参考重生

在工作流 01 或 03 的生成节点上设置 **重生轮次**，不需要增加节点或重新接线：

- `0`：只生成一次。
- `1–4`：首次生成后，再生成指定次数；`4` 共执行五次请求。
- 新版工作流 01、03 已设为 `4`，并开启 `single_chunk` 以对齐已验证的 Drama 3 短段实验；旧工作流省略轮次时默认 `0`，升级后可手动修改。
- 单音色参考全空时，仍只做一次普通生成。`reference_id` 不能用于此双参考路径，需要通过 Reference 提供原始音频。

每次重生始终保留当前音色的全部原始参考，再加入**上一轮生成音频**；不累计所有历史轮次。生成样本的参考原文取本段目标台词，去掉语气标签后送入；目标文稿本身仍保留标签。只有一份原始样本时，每次追加请求有两段参考；有多份原始样本时全部保留。

逐行模式在一行内完成所有轮次，再处理下一行；不同角色和不同行的生成参考互不传递。参考交换使用单声道 24-bit FLAC，中间响应使用非流式 WAV，避免反复有损编码。种子按已验证实验依次使用原值、加 1000、加 2000、加 3000，第 4 次重生沿用第 3 次种子。

输出指定轮次的最终结果，中间轮次不自动落盘。任一请求失败就停止，不自动补抽或退回较早结果；取消在请求边界检查，无法保证中断 Fish 已收到的推理。四轮意味着每行五次推理，耗时会增加，不能保证所有文本都随轮次增加而改善。插件未加入 ASR、漏读校正或自动情绪评分；生成样本若漏读，参考原文也不会被自动纠正。

工作流和接口也可用于 S2-Pro，但上述四轮偏好与音频复现是在 Drama 3 上验证的，S2-Pro 建议先设 `0` 做基线；其分块参数行为与 Drama 3 不同。长文本建议按情绪段落分行，并检查最终音频完整性。

### 参数与检查

同一个 **API 参数（自部署）** 节点可以连接三个生成节点。新版默认温度 `1.0`、Top P `0.9`、生成帧数上限 `1024`；重生轮次是插件本地控制项，不作为 API 字段发送。完整字段和模型适用范围见[插件 README](comfyui/fish_speech_local/README.md)与[API 参数说明](comfyui/fish_speech_local/docs/api-parameters.md)。

安装后可检查 Fish、插件注册和 ComfyUI CPU 状态：

```bash
cd /root/ComfyUI
source .comfy/bin/activate
python custom_nodes/fish_speech_local/tools/verify.py
```

加 `--generate` 会真实执行一次空参考普通生成。开发时可在安装了插件依赖和 CPU PyTorch 的环境中，从本仓库根目录运行现有测试：

```bash
FISH_SOURCE="$PWD" python -m unittest discover -s comfyui/fish_speech_local/tests -v
node comfyui/fish_speech_local/tests/test_parameters_ui.mjs
node comfyui/fish_speech_local/tests/test_reference_ui.mjs
```

实际 ComfyUI 执行测试需额外设置 `COMFY_SOURCE` 指向 ComfyUI 源码目录；否则会明确跳过。插件许可证保留在 [`comfyui/fish_speech_local/LICENSE`](comfyui/fish_speech_local/LICENSE)。

## 改造范围

当前专家卸载采用同步执行，以下列出实现范围和后续验证重点。

| 部分 | 实现范围 |
| --- | --- |
| 模型加载与显存分配 | 专家从加载开始就留在 CPU，避免先把完整模型装入 GPU |
| 专家缓存 | 每层固定配额，管理专家编号、缓存位置、淘汰和权重传输 |
| 专家执行 | 调入缺失专家后复用原 FP8 计算内核，保留路由权重和输出合并逻辑 |
| 预填充 | 在 MoE 内部按输入行分组，确保每组所需专家不超过缓存容量 |
| CUDA Graph | 卸载首版禁用原整步图捕获；正确性通过后再评估分段优化 |
| 请求执行 | 首版单 GPU 串行推理，保护共享状态与缓存位置的生命周期 |
| 日志 | 记录实际缓存大小、命中率、传输量、阶段耗时与显存峰值 |

实现主要位于 `fish_omni/backbone.py`、`fish_omni/moe.py` 和独立的 `fish_omni/expert_cache.py`，并接入模型初始化与生成入口。目录位于 `fish_speech/models/text2semantic/` 下。

首版不引入 CPU 专家计算、自定义 CUDA 内核、预测式预取或额外低位量化。这些优化应在基线验证后分别评估。

## 开发阶段与状态

| 阶段 | 完成条件 | 当前状态 |
| --- | --- | --- |
| 1. 项目说明 | 明确目标、边界、基线与验收方式 | 已整理 |
| 2. 加载与缓存 | CPU 专家存储、GPU 固定预算缓存、显式开关 | 已有实现，仍需目标 GPU 验收 |
| 3. 推理接入 | 预填充与逐帧生成正确执行，原路径可回退 | 已接入，原路径可通过开关选择 |
| 4. 自动验证 | 覆盖缓存命中、淘汰、重新载入和原路径回归 | 已有自动化测试与显式 GPU 验证工具 |
| 5. 真实 GPU 验收 | 完成 5090 32GB 的显存、速度、音频与连续运行测试 | 待执行 |
| 6. 性能优化 | 在正确性基线上评估异步传输与 CUDA Graph | 待评估 |

当前上游基线：[`541566932abbaebcb0a40e562a1f846738383f38`](https://github.com/fishaudio/fish-speech/commit/541566932abbaebcb0a40e562a1f846738383f38)。后续更新上游时，应记录新基线并重新执行相关验证。

## 资源预算与验证原则

目标运行环境为 Linux、NVIDIA RTX 5090 32GB，以及支持该显卡和原 FP8 内核的 PyTorch / CUDA / Triton 组合。具体依赖版本需结合实际镜像与宿主驱动验证。

初始规划如下，**均不是已经测得的最低配置或性能承诺**：

- GPU 专家缓存从 12 GiB 起步，验证后尝试 16 GiB；最终值由完整服务峰值决定。
- 初期预留约 3–4 GiB 显存余量，预算需包含参考编码、生成状态、临时工作区和音频解码。
- 优先使用 96GB 或以上 CPU 内存；64GB 是否足够，需核算加载峰值、容器额度及传输缓冲。
- 核对实际 PCIe 链路、内存带宽和容器资源限制；显卡计算能力不能代替专家传输性能。

验收分为三层：

1. **计算正确性**：在相同输入和历史下，对比原版与卸载版的路由、专家输出及 logits；检查专家权重与缩放参数的配对和字节完整性。
2. **任务与音频**：按相同 Case ID 验证中、英、日、法，覆盖无参考生成、声音克隆、语气、长文本和逐行多角色；检查漏读、静音、提前结束和音色稳定性。
3. **运行稳定性与性能**：区分冷启动与热缓存，记录端到端 RTF、预填充耗时、逐帧耗时、传输量及显存峰值；覆盖连续请求、取消恢复和长短请求切换。

保留原权重不保证同一种子输出逐样本一致。执行顺序与数值差异可能影响自回归生成，必须同时进行数值验证和实际音频评估。HTTP 200 或正常 EOS 也不能单独证明文稿完整生成。

## 安装与使用状态

专家卸载通过 Fish 进程的环境变量配置，默认关闭：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `FISH_DRAMA3_EXPERT_OFFLOAD` | `0` | `1` 开启 Drama 3 专家卸载，`0` 保留原版路径 |
| `FISH_DRAMA3_EXPERT_CACHE_GIB` | `12` | GPU 专家缓存预算，单位 GiB |
| `FISH_DRAMA3_EXPERT_RESERVE_GIB` | `4` | 显存余量预算，单位 GiB |
| `FISH_DRAMA3_EXPERT_PREFILL_ROWS` | `16` | 预填充阶段每批处理的输入行数 |

需要使用卸载时，在现有启动脚本启动 Fish 前设置这些变量。默认值不是 32GB 显卡的性能或显存保证；没有足够运行余量时不应盲目扩大缓存。已有验证工具为 `python -m tools.drama3_offload_verify --help`；实际 GPU 推理需显式使用 `--run`。插件本身无需启用卸载，也不负责配置专家缓存。

原版环境和接口说明可参考仓库内的[安装文档](docs/en/install.md)与[推理文档](docs/en/inference.md)。这些文档描述上游行为，包括原 Drama 3 的显存要求，不代表卸载模式已在所有显卡完成验收。已有可用镜像和模型可保留作对照。

## 上游与许可证

本项目是社区 Fork，不代表 Fish Audio 官方发布或支持。模型与基础推理实现来自 [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)。

保留仓库中的 [FISH AUDIO COMMUNITY LICENSE](LICENSE)；模型权重遵循其各自许可证。上游模型介绍及参考资料见[原中文说明](docs/README.zh.md)。

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Tech Report
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}

@misc{liao2026fishaudiodrama3technical,
      title={Fish Audio Drama 3 Technical Report}, 
      author={Dawei Han and Shidong Li and Shijia Liao and Jiahua Liu and Songting Liu and Qingzheng Wang and Yuxuan Wang and Xin Chen and Yifan Cheng and Tianyu Li and Gangin Park and Xuehao Yang and Chang Zeng and Haoyang Zhang and Ruoyi Zhang and Xiangyu Zhang},
      year={2026},
      eprint={2610.18010},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2610.18010}, 
}
```
