# MiniMax-M2 工具调用指南

[英文版](./tool_calling_guide.md) | [中文版](./tool_calling_guide_cn.md)

## 简介

MiniMax-M2 模型支持工具调用功能，使模型能够识别何时需要调用外部工具，并以结构化格式输出工具调用参数。本文档提供了有关如何使用 MiniMax-M2 工具调用功能的详细说明。

## 基础示例

以下 Python 脚本基于 OpenAI SDK 实现了一个天气查询工具调用示例：

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."

tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco? use celsius."}],
    tools=tools,
    tool_choice="auto"
)

print(response)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
```

**输出示例：**
```
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "celsius"}
Result: Getting the weather for San Francisco, CA in celsius...
```

## 手动解析模型输出

**我们强烈建议使用 vLLM 或 SGLnag 来解析工具调用。** 如果您无法使用支持 MiniMax-M2 的推理引擎（如 vLLM 和 SGLang）的内置解析器，或需要使用其他推理框架（如 transformers、TGI 等），您可以使用以下方法手动解析模型的原始输出。这种方法需要您自己解析模型输出的 XML 标签格式。

### 使用 Transformers 的示例

这是一个使用 transformers 库的完整示例：

```python
from transformers import AutoTokenizer

def get_default_tools():
    return [
        {
          "name": "get_current_weather",
          "description": "Get the latest weather for a location",
          "parameters": {
              "type": "object", 
              "properties": {
                  "location": {
                      "type": "string", 
                      "description": "A certain city, such as Beijing, Shanghai"
                  }
              }, 
          }
          "required": ["location"],
          "type": "object"
        }
    ]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

# Enable function calling tools
tools = get_default_tools()

# Apply chat template and include tool definitions
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# Send request (using any inference service)
import requests
payload = {
    "model": "MiniMaxAI/MiniMax-M2",
    "prompt": text,
    "max_tokens": 4096
}
response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Content-Type": "application/json"},
    json=payload,
    stream=False,
)

# Model output needs manual parsing
raw_output = response.json()["choices"][0]["text"]
print("Raw output:", raw_output)

# Use the parsing function below to process the output
tool_calls = parse_tool_calls(raw_output, tools)
```

## 🛠️ 工具调用定义

### 工具结构

工具调用需要在请求体中定义 `tools` 字段。每个工具由以下部分组成：

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "Search function.",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "Keywords for search, list should contain 1 element.",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "Category of query",
            "items": { "type": "string" },
            "type": "array"
          }
        },
        "required": [ "query_list", "query_tag" ],
        "type": "object"
      }
    }
  ]
}
```

**字段说明：**
- `name`：函数名称
- `description`：函数描述
- `parameters`：函数参数定义
  - `properties`：参数属性定义，其中键是参数名称，值包含详细的参数描述
  - `required`：必需参数列表
  - `type`：参数类型（通常为 "object"）

### 内部处理格式

在 MiniMax-M2 模型内部处理时，工具定义会被转换为特殊格式并连接到输入文本中。以下是一个完整示例：

```
]~!b[]~b]system
You are a helpful assistant.

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
<tool>{"name": "search_web", "description": "Search function.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}, "description": "Keywords for search, list should contain 1 element."}, "query_tag": {"type": "array", "items": {"type": "string"}, "description": "Category of query"}}, "required": ["query_list", "query_tag"]}}</tool>
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
[e~[
]~b]user
When were the latest announcements from OpenAI and Gemini?[e~[
]~b]ai
<think>
```

**格式说明：**

- `]~!b[]~b]system`：系统消息开始标记
- `[e~[`：消息结束标记
- `]~b]user`：用户消息开始标记
- `]~b]ai`：助手消息开始标记
- `]~b]tool`：工具结果消息开始标记
- `<tools>...</tools>`：工具定义区域，每个工具都用 `<tool>` 标签包装，内容为 JSON Schema
- `<minimax:tool_call>...</minimax:tool_call>`：工具调用区域
- `<think>...</think>`：生成过程中的思考过程标记

### 模型输出格式

MiniMax-M2 使用结构化的 XML 标签格式：

```xml
<minimax:tool_call>
<invoke name="search_web">
<parameter name="query_tag">["technology", "events"]</parameter>
<parameter name="query_list">["\"OpenAI\" \"latest\" \"release\""]</parameter>
</invoke>
<invoke name="search_web">
<parameter name="query_tag">["technology", "events"]</parameter>
<parameter name="query_list">["\"Gemini\" \"latest\" \"release\""]</parameter>
</invoke>
</minimax:tool_call>
```

每个工具调用使用 `<invoke name="function_name">` 标签，参数使用 `<parameter name="parameter_name">` 标签包装。

## 手动解析工具调用结果

### 解析工具调用

MiniMax-M2 使用结构化的 XML 标签，这需要一种不同的解析方法。核心函数如下：

```python
import re
import json
from typing import Any, Optional, List, Dict


def extract_name(name_str: str) -> str:
    """Extract name from quoted string"""
    name_str = name_str.strip()
    if name_str.startswith('"') and name_str.endswith('"'):
        return name_str[1:-1]
    elif name_str.startswith("'") and name_str.endswith("'"):
        return name_str[1:-1]
    return name_str


def convert_param_value(value: str, param_type: str) -> Any:
    """Convert parameter value based on parameter type"""
    if value.lower() == "null":
        return None
        
    param_type = param_type.lower()
    
    if param_type in ["string", "str", "text"]:
        return value
    elif param_type in ["integer", "int"]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    elif param_type in ["number", "float"]:
        try:
            val = float(value)
            return val if val != int(val) else int(val)
        except (ValueError, TypeError):
            return value
    elif param_type in ["boolean", "bool"]:
        return value.lower() in ["true", "1"]
    elif param_type in ["object", "array"]:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    else:
        # Try JSON parsing, return string if failed
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def parse_tool_calls(model_output: str, tools: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract all tool calls from model output
    
    Args:
        model_output: Complete output text from the model
        tools: Tool definition list for getting parameter type information, format can be:
               - [{"name": "...", "parameters": {...}}]
               - [{"type": "function", "function": {"name": "...", "parameters": {...}}}]
    
    Returns:
        Parsed tool call list, each element contains name and arguments fields
    
    Example:
        >>> tools = [{
        ...     "name": "get_weather",
        ...     "parameters": {
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {"type": "string"},
        ...             "unit": {"type": "string"}
        ...         }
        ...     }
        ... }]
        >>> output = '''<minimax:tool_call>
        ... <invoke name="get_weather">
        ... <parameter name="location">San Francisco</parameter>
        ... <parameter name="unit">celsius</parameter>
        ... </invoke>
        ... </minimax:tool_call>'''
        >>> result = parse_tool_calls(output, tools)
        >>> print(result)
        [{'name': 'get_weather', 'arguments': {'location': 'San Francisco', 'unit': 'celsius'}}]
    """
    # Quick check if tool call marker is present
    if "<minimax:tool_call>" not in model_output:
        return []
    
    tool_calls = []
    
    try:
        # Match all <minimax:tool_call> blocks
        tool_call_regex = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
        parameter_regex = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)
        
        # Iterate through all tool_call blocks
        for tool_call_match in tool_call_regex.findall(model_output):
            # Iterate through all invokes in this block
            for invoke_match in invoke_regex.findall(tool_call_match):
                # Extract function name
                name_match = re.search(r'^([^>]+)', invoke_match)
                if not name_match:
                    continue
                
                function_name = extract_name(name_match.group(1))
                
                # Get parameter configuration
                param_config = {}
                if tools:
                    for tool in tools:
                        tool_name = tool.get("name") or tool.get("function", {}).get("name")
                        if tool_name == function_name:
                            params = tool.get("parameters") or tool.get("function", {}).get("parameters")
                            if isinstance(params, dict) and "properties" in params:
                                param_config = params["properties"]
                            break
                
                # Extract parameters
                param_dict = {}
                for match in parameter_regex.findall(invoke_match):
                    param_match = re.search(r'^([^>]+)>(.*)', match, re.DOTALL)
                    if param_match:
                        param_name = extract_name(param_match.group(1))
                        param_value = param_match.group(2).strip()
                        
                        # Remove leading and trailing newlines
                        if param_value.startswith('\n'):
                            param_value = param_value[1:]
                        if param_value.endswith('\n'):
                            param_value = param_value[:-1]
                        
                        # Get parameter type and convert
                        param_type = "string"
                        if param_name in param_config:
                            if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
                                param_type = param_config[param_name]["type"]
                        
                        param_dict[param_name] = convert_param_value(param_value, param_type)
                
                tool_calls.append({
                    "name": function_name,
                    "arguments": param_dict
                })
    
    except Exception as e:
        print(f"Failed to parse tool calls: {e}")
        return []
    
    return tool_calls
```

**使用示例：**

```python
# Define tools
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            },
            "required": ["location", "unit"]
        }
    }
]

# Model output
model_output = """Let me help you query the weather.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"""

# Parse tool calls
tool_calls = parse_tool_calls(model_output, tools)

# Output results
for call in tool_calls:
    print(f"Function called: {call['name']}")
    print(f"Arguments: {call['arguments']}")
    # Output: Function called: get_weather
    #         Arguments: {'location': 'San Francisco', 'unit': 'celsius'}
```

### 执行工具调用

完成解析后，您可以执行相应的工具并构造返回结果：

```python
def execute_function_call(function_name: str, arguments: dict):
    """Execute function call and return result"""
    if function_name == "get_weather":
        location = arguments.get("location", "Unknown location")
        unit = arguments.get("unit", "celsius")
        # Build function execution result
        return {
            "role": "tool", 
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": json.dumps({
                    "location": location, 
                    "temperature": "25", 
                    "unit": unit, 
                    "weather": "Sunny"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # Simulate search results
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"Search keywords: {query_list}, Category: {query_tag}\nSearch results: Relevant information found"
              }
            ]
          }
    
    return None
```

### 将工具执行结果返回给模型

在成功解析工具调用后，您应该将工具执行结果添加到对话历史中，以便模型在后续交互中可以访问和利用这些信息。请参考 [chat_template.jinja](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/chat_template.jinja) 了解连接格式。

## 参考文献

- [MiniMax-M2 模型仓库](https://github.com/MiniMax-AI/MiniMax-M2)
- [vLLM 项目主页](https://github.com/vllm-project/vllm)
- [SGLang 项目主页](https://github.com/sgl-project/sglang)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

## 获取支持

如果遇到任何问题：

- 通过邮箱 [model@minimax.io](mailto:model@minimax.io) 等官方渠道联系我们的技术支持团队

- 在我们的仓库提交 Issue

- 通过我们的[M2企业微信群](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/figures/wechat.jpeg) 反馈

我们会持续优化模型的使用体验，欢迎反馈！