# MiniMax-M2 Tool Calling Guide

## Introduction

The MiniMax-M2 model supports tool calling capabilities, enabling the model to identify when external tools need to be called and output tool call parameters in a structured format. This document provides detailed instructions on how to use the tool calling features of MiniMax-M2.

## Basic Example

The following Python script implements a weather query tool call example based on the OpenAI SDK:

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

**Output Example:**
```
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "celsius"}
Result: Getting the weather for San Francisco, CA in celsius...
```

## Manually Parsing Model Output

**We strongly recommend using vLLM or SGLang for parsing tool calls.** If you cannot use the built-in parser of inference engines (e.g., vLLM and SGLang) that support MiniMax-M2, or need to use other inference frameworks (such as transformers, TGI, etc.), you can manually parse the model's raw output using the following method. This approach requires you to parse the XML tag format of the model output yourself.

### Example Using Transformers

Here is a complete example using the transformers library:

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

## 🛠️ Tool Call Definition

### Tool Structure

Tool calls need to define the `tools` field in the request body. Each tool consists of the following parts:

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

**Field Descriptions:**
- `name`: Function name
- `description`: Function description
- `parameters`: Function parameter definition
  - `properties`: Parameter property definition, where key is the parameter name and value contains detailed parameter description
  - `required`: List of required parameters
  - `type`: Parameter type (usually "object")

### Internal Processing Format

When processing within the MiniMax-M2 model, tool definitions are converted to a special format and concatenated to the input text. Here is a complete example:

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

**Format Description:**

- `]~!b[]~b]system`: System message start marker
- `[e~[`: Message end marker
- `]~b]user`: User message start marker
- `]~b]ai`: Assistant message start marker
- `]~b]tool`: Tool result message start marker
- `<tools>...</tools>`: Tool definition area, each tool is wrapped with `<tool>` tag, content is JSON Schema
- `<minimax:tool_call>...</minimax:tool_call>`: Tool call area
- `<think>...</think>`: Thinking process marker during generation

### Model Output Format

MiniMax-M2 uses structured XML tag format:

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

Each tool call uses the `<invoke name="function_name">` tag, and parameters use the `<parameter name="parameter_name">` tag wrapper.

## Manually Parsing Tool Call Results

### Parsing Tool Calls

MiniMax-M2 uses structured XML tags, which require a different parsing approach. The core function is as follows:

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

**Usage Example:**

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

### Executing Tool Calls

After parsing is complete, you can execute the corresponding tool and construct the return result:

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

### Returning Tool Execution Results to the Model

After successfully parsing tool calls, you should add the tool execution results to the conversation history so that the model can access and utilize this information in subsequent interactions. Refer to [chat_template.jinja](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/chat_template.jinja) for concatenation format.

## References

- [MiniMax-M2 Model Repository](https://github.com/MiniMax-AI/MiniMax-M2)
- [vLLM Project Homepage](https://github.com/vllm-project/vllm)
- [SGLang Project Homepage](https://github.com/sgl-project/sglang)
- [OpenAI Python SDK](https://github.com/openai/openai-python)