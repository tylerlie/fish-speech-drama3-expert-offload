# IndexTTS Local 0.1.0

独立的 CPU ComfyUI 客户端，连接本仓库 `deployment/a10g_compare/index_api.py` HTTP 适配器。支持 IndexTTS 2.0、2.5，模型运行在独立 GPU 服务。不能直接连接任意同名第三方 API。

安装到 `ComfyUI/custom_nodes/indextts_local`，在 ComfyUI 环境执行 `python -m pip install -r requirements.txt`，重启并导入 `workflows/01_IndexTTS_克隆与默认音色生成.json`。

服务根地址通过 `INDEX_TTS_API_URL` 设置，默认 `http://127.0.0.1:18081`。

- `Index Reference`：本节点上传或选择已有媒体，也支持上游 AUDIO 连线，连线优先。备注可空且不送入推理；Index 不需要参考原文。
- `Index 参数`：reference / text / vector / audio 四种情绪模式，只启用对应模式。vector 的八个数值经官方方法归一化。2.0 不支持非 1 的 duration_factor；2.5 支持。
- `Index 克隆 / 默认音色生成`：音色 Reference 可空，使用服务端默认参考。**不是真正的无参考生成**。标准 AUDIO 输出可接原生试听/保存节点。

所有输入节点均允许留空，不需要删除闲置情绪 Reference。只有选择 audio 情绪模式而未提供情绪音频时会报错。有音色参考则使用上传音色，空则使用服务器默认音色；不会悄悄忽略损坏的非空音频。

本节点一次请求、一次推理，没有 Fish 插件的多轮重生。生成正文不要包含 Fish 专有情绪标签；情绪要求放独立参数中。2.0/2.5 切换由后台启动脚本控制，同一个工作流不需改线。

完整 Linux 部署步骤见仓库 `deployment/a10g_compare/README.md`。本地模拟模型与 CPU ComfyUI 验证不替代真实 A10G 验收。
