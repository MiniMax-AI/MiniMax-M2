# MiniMax M2 模型 vLLM 部署指南

我们推荐使用 [vLLM](https://docs.vllm.ai/en/stable/) 来部署 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 模型。vLLM 是一个高性能的推理引擎，其具有卓越的服务吞吐、高效智能的内存管理机制、强大的批量请求处理能力、深度优化的底层性能等特性。我们建议在部署之前查看 vLLM 的官方文档以检查硬件兼容性。

## 环境要求

- OS：Linux

- Python：3.9 - 3.12

- GPU：

  - compute capability 7.0 or higher

  - 显存需求：权重需要 220 GB，每 1M 上下文 token 需要 60 GB

以下为推荐配置，实际需求请根据业务场景调整：

- 96G x4 GPU：支持 40 万 token 的上下文输入。

- 144G x8 GPU：支持长达 300 万 token 的上下文输入。

## 使用 Python 部署

建议使用虚拟环境（如 venv、conda、uv）以避免依赖冲突。建议在全新的 Python 环境中安装 vLLM：
```bash
# 尚未 release，请安装 nightly 构建
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
# 如果 release，使用 uv 安装
uv pip install "vllm" --torch-backend=auto
```

运行如下命令启动 vLLM 服务器，vLLM 会自动从 Huggingface 下载并缓存 MiniMax-M2 模型。

4 卡部署命令：

```bash
SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=0 vllm serve \
    --model MiniMaxAI/MiniMax-M2 \
    --trust-remote-code \
    --enable-expert-parallel --tensor-parallel-size 4 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2
```

## 测试部署

启动后，可以通过如下命令测试 vLLM OpenAI 兼容接口：

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

该 vLLM 版本过旧，请升级到最新版本。

## 获取支持

如果在部署 MiniMax 模型过程中遇到任何问题：

- 通过邮箱 api@minimaxi.com 等官方渠道联系我们的技术支持团队

- 在我们的 [GitHub](https://github.com/MiniMax-AI) 仓库提交 Issue
我们会持续优化模型的部署体验，欢迎反馈！
