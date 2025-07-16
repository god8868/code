# AI模型适配器

一个统一的AI模型调用接口，支持多种主流大语言模型提供商，包括OpenAI、Claude、Gemini等。

## 特性

- 🚀 **统一接口**: 使用相同的API调用不同的AI模型
- 🔌 **多提供商支持**: 支持OpenAI、Claude、Gemini等主流模型
- ⚙️ **灵活配置**: 支持配置文件和环境变量配置
- 🔄 **流式输出**: 支持流式文本生成
- 📊 **模型比较**: 轻松比较不同模型的响应
- 🏥 **健康检查**: 内置模型健康检查功能
- 📈 **使用统计**: 自动记录token使用量和延迟

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置API密钥

设置环境变量：

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

### 2. 基本使用

```python
from ai_model_adapter import ModelManager

# 创建模型管理器
manager = ModelManager()

# 生成文本
response = manager.generate(
    "请介绍一下人工智能",
    model_name="gpt-3.5-turbo"
)

print(f"响应: {response.content}")
print(f"用量: {response.usage}")
print(f"延迟: {response.latency:.2f}秒")
```

### 3. 对话模式

```python
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助您的吗？"},
    {"role": "user", "content": "请介绍一下Python"}
]

response = manager.chat(messages, model_name="claude-3-sonnet")
print(response.content)
```

### 4. 流式生成

```python
for chunk in manager.stream_generate(
    "请介绍一下机器学习",
    model_name="gpt-4"
):
    print(chunk, end="")
```

## 配置文件

系统会自动创建 `ai_model_config.yaml` 配置文件：

```yaml
default_model: gpt-3.5-turbo
models:
  gpt-3.5-turbo:
    model_type: openai
    model_name: gpt-3.5-turbo
    api_key_env: OPENAI_API_KEY
    max_tokens: 4000
    temperature: 0.7
    timeout: 30
    extra_params: {}
  
  claude-3-sonnet:
    model_type: claude
    model_name: claude-3-sonnet-20240229
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4000
    temperature: 0.7
    timeout: 30
    extra_params: {}
  
  gemini-pro:
    model_type: gemini
    model_name: gemini-pro
    api_key_env: GOOGLE_API_KEY
    max_tokens: 4000
    temperature: 0.7
    timeout: 30
    extra_params: {}
```

## 高级功能

### 动态添加模型

```python
from ai_model_adapter import ModelConfig, ModelType

# 添加自定义模型配置
config = ModelConfig(
    model_type=ModelType.OPENAI,
    model_name="gpt-4o",
    api_key="your_api_key",
    temperature=0.5,
    max_tokens=2000
)

manager.add_model("my-gpt4", config)
```

### 模型比较

```python
# 比较多个模型的响应
comparison = manager.compare_models(
    "请解释什么是深度学习",
    ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]
)

for model_name, response in comparison.items():
    print(f"{model_name}: {response.content}")
```

### 批量生成

```python
prompts = [
    "什么是人工智能？",
    "什么是机器学习？",
    "什么是深度学习？"
]

results = manager.batch_generate(prompts, model_name="gpt-3.5-turbo")
for i, response in enumerate(results):
    print(f"问题{i+1}: {response.content}")
```

### 健康检查

```python
health = manager.health_check("gpt-3.5-turbo")
print(f"状态: {health['status']}")
if health['status'] == 'healthy':
    print(f"延迟: {health['latency']:.2f}秒")
```

## 支持的模型

### OpenAI
- GPT-4, GPT-4 Turbo
- GPT-3.5 Turbo
- GPT-4o

### Claude (Anthropic)
- Claude 3.5 Sonnet
- Claude 3.5 Haiku
- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku

### Gemini (Google)
- Gemini Pro
- Gemini Pro Vision
- 其他Gemini模型

## 扩展适配器

您可以轻松添加对新模型的支持：

```python
from ai_model_adapter import BaseAdapter, ModelType

class CustomAdapter(BaseAdapter):
    def _validate_config(self):
        # 验证配置逻辑
        pass
    
    def generate(self, prompt, **kwargs):
        # 实现生成逻辑
        pass
    
    def chat(self, messages, **kwargs):
        # 实现对话逻辑
        pass
    
    def stream_generate(self, prompt, **kwargs):
        # 实现流式生成逻辑
        pass
    
    def stream_chat(self, messages, **kwargs):
        # 实现流式对话逻辑
        pass

# 注册自定义适配器
manager.register_adapter(ModelType.CUSTOM, CustomAdapter)
```

## 环境变量

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API密钥 |
| `ANTHROPIC_API_KEY` | Anthropic API密钥 |
| `GOOGLE_API_KEY` | Google API密钥 |

## 错误处理

系统提供了完善的错误处理机制：

```python
try:
    response = manager.generate("Hello", model_name="non-existent-model")
except ValueError as e:
    print(f"配置错误: {e}")
except RuntimeError as e:
    print(f"API调用错误: {e}")
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持OpenAI、Claude、Gemini模型
- 统一的API接口
- 配置文件支持
- 流式输出支持
