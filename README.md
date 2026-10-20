# Fish Speech · Drama 3 Expert Offload

基于 [Fish Speech 官方仓库](https://github.com/fishaudio/fish-speech)的社区改造项目，目标是在 **单张 RTX 5090 32GB** 上运行 Drama 3，保留官方 Fish API，并继续使用现有 ComfyUI 插件和工作流。

改造方向是将路由专家的原始 FP8 权重保存在 CPU 内存中，使用固定预算的 GPU 专家缓存按需调入并执行。其余模型部分优先常驻 GPU，不额外进行 INT4 量化，不裁剪专家，也不减少原模型每次选择的专家数量。

> **当前状态：方案确定，尚未实现专家卸载。** 当前推理代码仍为上游实现，不能直接视为支持 32GB 显卡。本文中的卸载能力、资源预算和兼容范围是开发目标；速度、音质与显存峰值需要真实 GPU 验证。

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

优先验收现有三类工作流；插件和工作流文件由现有 ComfyUI 项目提供，本仓库负责语音后端：

| 工作流 | 输入与行为 | 改造后的验收重点 |
| --- | --- | --- |
| 单音色克隆 / 生成 | 参考音频与对应文本可空；为空时生成语音 | 有参考与无参考两条路径 |
| 原生多角色长文本 | 沿用现有角色集合与文本格式 | 接口回归；现有漏读问题单独记录 |
| 拼接式多角色多语气 | 按非空换行逐次发送单人克隆请求，最多 9 个克隆角色 | 角色切换、语气、完整拼接与音频列表 |

逐行循环可以复用同一份模型与专家缓存，无需为每个角色加载一份模型。请求的参考信息、生成状态和随机种子必须保持隔离。专家卸载本身不意味着原生多角色漏读已修复。

## 改造范围

第一版先采用同步执行，建立可验证的正确性基线。

| 部分 | 计划改动 |
| --- | --- |
| 模型加载与显存分配 | 专家从加载开始就留在 CPU，避免先把完整模型装入 GPU |
| 专家缓存 | 每层固定配额，管理专家编号、缓存位置、淘汰和权重传输 |
| 专家执行 | 调入缺失专家后复用原 FP8 计算内核，保留路由权重和输出合并逻辑 |
| 预填充 | 在 MoE 内部按输入行分组，确保每组所需专家不超过缓存容量 |
| CUDA Graph | 卸载首版禁用原整步图捕获；正确性通过后再评估分段优化 |
| 请求执行 | 首版单 GPU 串行推理，保护共享状态与缓存位置的生命周期 |
| 日志 | 记录实际缓存大小、命中率、传输量、阶段耗时与显存峰值 |

预计主要涉及 `fish_omni/backbone.py`、`fish_omni/moe.py`，新增独立缓存模块，并按需要调整模型初始化与生成入口。目录位于 `fish_speech/models/text2semantic/` 下。

首版不引入 CPU 专家计算、自定义 CUDA 内核、预测式预取或额外低位量化。这些优化应在基线验证后分别评估。

## 开发阶段与状态

| 阶段 | 完成条件 | 当前状态 |
| --- | --- | --- |
| 1. 项目说明 | 明确目标、边界、基线与验收方式 | 已整理 |
| 2. 加载与缓存 | CPU 专家存储、GPU 固定预算缓存、显式开关 | 待实现 |
| 3. 推理接入 | 预填充与逐帧生成正确执行，原路径可回退 | 待实现 |
| 4. 自动验证 | 覆盖缓存命中、淘汰、重新载入和原路径回归 | 待执行 |
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

当前尚无可用的专家卸载启动参数或 5090 专用启动脚本；不要仅替换仓库就认为已经完成 32GB 适配。参数与操作步骤将在实现并验证后补充。

原版环境和接口说明可参考仓库内的[安装文档](docs/en/install.md)与[推理文档](docs/en/inference.md)。这些文档当前仍描述上游行为，包括原 Drama 3 的显存要求，不代表本项目已完成适配。已有可用镜像和模型可保留作对照，无需在文档准备阶段重装。

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
