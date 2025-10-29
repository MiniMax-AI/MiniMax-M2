# MiniMax M2 模型 SGLang 部署指南

[英文版](./sglang_deploy_guide.md) | [中文版](./sglang_deploy_guide_cn.md)

我们推荐使用 [SGLang](https://github.com/sgl-project/sglang) 来部署 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 模型。SGLang 是一个高性能的推理引擎，其具有卓越的服务吞吐、高效智能的内存管理机制、强大的批量请求处理能力、深度优化的底层性能等特性。我们建议在部署之前查看 SGLang 的官方文档以检查硬件兼容性。

## 本文档适用模型

本文档适用以下模型，只需在部署时修改模型名称即可。

- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

以下以 MiniMax-M2 为例说明部署流程。

## 环境要求

- OS：Linux

- Python：3.9 - 3.12

- GPU：

  - compute capability 7.0 or higher

  - 显存需求：权重需要 220 GB，每 1M 上下文 token 需要 240 GB

以下为推荐配置，实际需求请根据业务场景调整：

- 96G x4 GPU：支持 40 万 token 的总上下文。

- 144G x8 GPU：支持长达 300 万 token 的总上下文。

## 使用 Python 部署

建议使用虚拟环境（如 **venv**、**conda**、**uv**）以避免依赖冲突。

建议在全新的 Python 环境中安装 SGLang:
```bash
git clone -b v0.5.4.post1 https://github.com/sgl-project/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

运行如下命令启动 SGLang 服务器，SGLang 会自动从 Huggingface 下载并缓存 MiniMax-M2 模型。

4 卡部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

8 卡部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --reasoning-parser minimax-append-think \
    --port 8000 \
    --mem-fraction-static 0.85
```

## 测试部署

启动后，可以通过如下命令测试 SGLang OpenAI 兼容接口：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```

## 常见问题

### Huggingface 网络问题

如果遇到网络问题，可以设置代理后再进行拉取。

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### MiniMax-M2 model is not currently supported

请升级到最新的稳定版本, >= v0.5.4.post1.

## 获取支持

如果在部署 MiniMax 模型过程中遇到任何问题：

- 通过邮箱 [model@minimax.io](mailto:model@minimax.io) 等官方渠道联系我们的技术支持团队

- 在我们的 [GitHub](https://github.com/MiniMax-AI) 仓库提交 Issue

- 通过我们的 [官方企业微信交流群](https://github.com/MiniMax-AI/MiniMax-AI.github.io/blob/main/images/wechat-qrcode.jpeg) 反馈

我们会持续优化模型的部署体验，欢迎反馈！
