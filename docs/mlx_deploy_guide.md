## MLX deployment guide

Run, serve, and fine-tune [**MiniMax-M2**](https://huggingface.co/MiniMaxAI/MiniMax-M2) locally on your Mac using the **MLX** framework. This guide gets you up and running quickly.

> **Requirements**  
> - Apple Silicon Mac (M3 Ultra or later)  
> - **At least 256GB of unified memory (RAM)**  


**Installation**

Install the `mlx-lm` package via pip:

```bash
pip install mlx-lm
```

**CLI**

Generate text directly from the terminal:

```bash
mlx_lm.generate \
  --model mlx-community/MiniMax-M2-4bit \
  --prompt "How tall is Mount Everest?"
```

> Add `--max-tokens 256` to control response length, or `--temp 0.7` for creativity.

**Python Script Example**

Use `mlx-lm` in your own Python scripts:

```python
from mlx_lm import load, generate

# Load the quantized model
model, tokenizer = load("mlx-community/MiniMax-M2-4bit")

prompt = "Hello, how are you?"

# Apply chat template if available (recommended for chat models)
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Generate response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=256,
    temp=0.7,
    verbose=True
)

print(response)
```

**Tips**
- **Model variants**: Check [Hugging Face](https://huggingface.co/collections/mlx-community/minimax-m2) for `MiniMax-M2-4bit`, `6bit`, `8bit`, or `bfloat16` versions.
- **Fine-tuning**: Use `mlx-lm.lora` for efficient parameter-efficient fine-tuning (PEFT).

**Resources**  
- GitHub: [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  
- Models: [https://huggingface.co/mlx-community](https://huggingface.co/mlx-community)