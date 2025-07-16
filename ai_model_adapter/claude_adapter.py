"""
Claude (Anthropic) 适配器实现
"""

import time
from typing import Dict, Any, List, Generator
from .base import BaseAdapter, ModelResponse, ModelConfig, ModelType


class ClaudeAdapter(BaseAdapter):
    """Claude适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _validate_config(self) -> None:
        """验证配置"""
        if self.config.model_type != ModelType.CLAUDE:
            raise ValueError(f"Expected Claude model type, got {self.config.model_type}")
        if not self.config.api_key:
            raise ValueError("Claude API key is required")
    
    def _init_client(self):
        """初始化Claude客户端"""
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """对话模式"""
        start_time = time.time()
        
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            **self.config.extra_params,
            **kwargs
        }
        
        try:
            response = self._client.messages.create(**params)
            latency = time.time() - start_time
            
            # 提取响应内容
            content = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            } if response.usage else None
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                usage=usage,
                finish_reason=response.stop_reason,
                latency=latency,
                metadata={"response_id": response.id}
            )
        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {str(e)}")
    
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        messages = [{"role": "user", "content": prompt}]
        yield from self.stream_chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
            **self.config.extra_params,
            **kwargs
        }
        
        try:
            stream = self._client.messages.create(**params)
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
        except Exception as e:
            raise RuntimeError(f"Claude stream API call failed: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        # Claude API 没有直接的模型列表接口，返回常见模型
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]