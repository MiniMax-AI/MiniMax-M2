# MiniMax-M2 函数调用（Function Call）功能指南

## 简介

MiniMax-M2 模型支持函数调用功能，使模型能够识别何时需要调用外部函数，并以结构化格式输出函数调用参数。本文档详细介绍了如何使用 MiniMax-M2 的函数调用功能。

## 基础示例

以下 Python 脚本基于 OpenAI SDK 实现了一个天气查询函数的调用示例：

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

如果您无法使用已支持 MiniMax-M2 的推理引擎的内置解析器，或者需要使用其他推理框架（如 transformers、TGI 等），可以使用以下方法手动解析模型的原始输出。这种方法需要您自己解析模型输出的 XML 标签格式。

### 使用 Transformers 的示例

以下是使用 transformers 库的完整示例：

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

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

# 启用函数调用工具
tools = get_default_tools()

# 应用聊天模板，并加入工具定义
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# 发送请求（这里使用任何推理服务）
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

# 模型输出需要手动解析
raw_output = response.json()["choices"][0]["text"]
print("原始输出:", raw_output)

# 使用下面的解析函数处理输出
function_calls = parse_tool_calls(raw_output, tools)
```

## 🛠️ 函数调用的定义

### 函数结构体

函数调用需要在请求体中定义 `tools` 字段，每个函数由以下部分组成：

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "搜索函数。",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "进行搜索的关键词，列表元素个数为1。",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "query的分类",
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
- `name`: 函数名称
- `description`: 函数功能描述
- `parameters`: 函数参数定义
  - `properties`: 参数属性定义，key 是参数名，value 包含参数的详细描述
  - `required`: 必填参数列表
  - `type`: 参数类型（通常为 "object"）

### 模型内部处理格式

在 MiniMax-M2 模型内部处理时，函数定义会被转换为特殊格式并拼接到输入文本中。以下是一个完整的示例：

```
]~!b[]~b]system
You are a helpful assistant.

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
<tool>{"name": "search_web", "description": "搜索函数。", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}, "description": "进行搜索的关键词，列表元素个数为1。"}, "query_tag": {"type": "array", "items": {"type": "string"}, "description": "query的分类"}}, "required": ["query_list", "query_tag"]}}</tool>
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
OpenAI 和 Gemini 的最近一次发布会都是什么时候?[e~[
]~b]ai
<think>
```

**格式说明：**

- `]~!b[]~b]system`: System 消息开始标记
- `[e~[`: 消息结束标记
- `]~b]user`: User 消息开始标记
- `]~b]ai`: Assistant 消息开始标记
- `]~b]tool`: Tool 结果消息开始标记
- `<tools>...</tools>`: 工具定义区域，每个工具用 `<tool>` 标签包裹，内容为 JSON Schema
- `<minimax:tool_call>...</minimax:tool_call>`: 工具调用区域
- `<think>`: 生成时的思考过程标记（可选）

### 模型输出格式

MiniMax-M2使用结构化的 XML 标签格式：

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

每个函数调用使用 `<invoke name="函数名">` 标签，参数使用 `<parameter name="参数名">` 标签包裹。

## 手动解析函数调用结果

### 解析函数调用

MiniMax-M2使用结构化的 XML 标签，需要不同的解析方式。核心函数如下：

```python
import re
import json
from typing import Any, Optional, List, Dict


def extract_name(name_str: str) -> str:
    """从引号包裹的字符串中提取名称"""
    name_str = name_str.strip()
    if name_str.startswith('"') and name_str.endswith('"'):
        return name_str[1:-1]
    elif name_str.startswith("'") and name_str.endswith("'"):
        return name_str[1:-1]
    return name_str


def convert_param_value(value: str, param_type: str) -> Any:
    """根据参数类型转换参数值"""
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
        # 尝试 JSON 解析，失败则返回字符串
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def parse_tool_calls(model_output: str, tools: Optional[List[Dict]] = None) -> List[Dict]:
    """
    从模型输出中提取所有工具调用
    
    Args:
        model_output: 模型的完整输出文本
        tools: 工具定义列表，用于获取参数类型信息，格式可以是：
               - [{"name": "...", "parameters": {...}}]
               - [{"type": "function", "function": {"name": "...", "parameters": {...}}}]
    
    Returns:
        解析后的工具调用列表，每个元素包含 name 和 arguments 字段
    
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
    # 快速检查是否包含工具调用标记
    if "<minimax:tool_call>" not in model_output:
        return []
    
    tool_calls = []
    
    try:
        # 匹配所有 <minimax:tool_call> 块
        tool_call_regex = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
        parameter_regex = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)
        
        # 遍历所有 tool_call 块
        for tool_call_match in tool_call_regex.findall(model_output):
            # 遍历该块中的所有 invoke
            for invoke_match in invoke_regex.findall(tool_call_match):
                # 提取函数名
                name_match = re.search(r'^([^>]+)', invoke_match)
                if not name_match:
                    continue
                
                function_name = extract_name(name_match.group(1))
                
                # 获取参数配置
                param_config = {}
                if tools:
                    for tool in tools:
                        tool_name = tool.get("name") or tool.get("function", {}).get("name")
                        if tool_name == function_name:
                            params = tool.get("parameters") or tool.get("function", {}).get("parameters")
                            if isinstance(params, dict) and "properties" in params:
                                param_config = params["properties"]
                            break
                
                # 提取参数
                param_dict = {}
                for match in parameter_regex.findall(invoke_match):
                    param_match = re.search(r'^([^>]+)>(.*)', match, re.DOTALL)
                    if param_match:
                        param_name = extract_name(param_match.group(1))
                        param_value = param_match.group(2).strip()
                        
                        # 去除首尾的换行符
                        if param_value.startswith('\n'):
                            param_value = param_value[1:]
                        if param_value.endswith('\n'):
                            param_value = param_value[:-1]
                        
                        # 获取参数类型并转换
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
        print(f"解析工具调用失败: {e}")
        return []
    
    return tool_calls
```

**使用示例：**

```python
# 定义工具
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

# 模型输出
model_output = """我来帮你查询天气。
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"""

# 解析工具调用
tool_calls = parse_tool_calls(model_output, tools)

# 输出结果
for call in tool_calls:
    print(f"调用函数: {call['name']}")
    print(f"参数: {call['arguments']}")
    # 输出: 调用函数: get_weather
    #      参数: {'location': 'San Francisco', 'unit': 'celsius'}
```

### 执行函数调用

解析完成后，您可以执行对应的函数并构建返回结果：

```python
def execute_function_call(function_name: str, arguments: dict):
    """执行函数调用并返回结果"""
    if function_name == "get_weather":
        location = arguments.get("location", "未知位置")
        unit = arguments.get("unit", "celsius")
        # 构建函数执行结果
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
                    "weather": "晴朗"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # 模拟搜索结果
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"搜索关键词: {query_list}, 分类: {query_tag}\n搜索结果: 相关信息已找到"
              }
            ]
          }
    
    return None
```

### 将函数执行结果返回给模型

成功解析函数调用后，您应将函数执行结果添加到对话历史中，以便模型在后续交互中能够访问和利用这些信息，拼接格式参考chat_template.jinja

## 参考资料

- [MiniMax-M2 模型仓库](https://github.com/MiniMax-AI/MiniMax-M2)
- [vLLM 项目主页](https://github.com/vllm-project/vllm)
- [OpenAI Python SDK](https://github.com/openai/openai-python)