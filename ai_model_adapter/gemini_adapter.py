"""
Gemini (Google) 适配器实现
"""

import time
from typing import Dict, Any, List, Generator
from .base import BaseAdapter, ModelResponse, ModelConfig, ModelType


class GeminiAdapter(BaseAdapter):
    """Gemini适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _validate_config(self) -> None:
        """验证配置"""
        if self.config.model_type != ModelType.GEMINI:
            raise ValueError(f"Expected Gemini model type, got {self.config.model_type}")
        if not self.config.api_key:
            raise ValueError("Gemini API key is required")
    
    def _init_client(self):
        """初始化Gemini客户端"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    **self.config.extra_params
                )
            )
        except ImportError:
            raise ImportError("Please install google-generativeai package: pip install google-generativeai")
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """生成文本"""
        start_time = time.time()
        
        try:
            response = self._client.generate_content(
                prompt,
                generation_config=self._get_generation_config(kwargs)
            )
            latency = time.time() - start_time
            
            # 提取响应内容
            content = response.text if response.text else ""
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            } if response.usage_metadata else None
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                usage=usage,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                latency=latency,
                metadata={"safety_ratings": response.candidates[0].safety_ratings if response.candidates else None}
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """对话模式"""
        start_time = time.time()
        
        # 转换消息格式
        history = []
        for msg in messages[:-1]:  # 除了最后一条消息
            role = "user" if msg["role"] == "user" else "model"
            history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        try:
            chat = self._client.start_chat(history=history)
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=self._get_generation_config(kwargs)
            )
            latency = time.time() - start_time
            
            # 提取响应内容
            content = response.text if response.text else ""
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            } if response.usage_metadata else None
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                usage=usage,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                latency=latency,
                metadata={"safety_ratings": response.candidates[0].safety_ratings if response.candidates else None}
            )
        except Exception as e:
            raise RuntimeError(f"Gemini chat API call failed: {str(e)}")
    
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        try:
            response = self._client.generate_content(
                prompt,
                generation_config=self._get_generation_config(kwargs),
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(f"Gemini stream API call failed: {str(e)}")
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        # 转换消息格式
        history = []
        for msg in messages[:-1]:  # 除了最后一条消息
            role = "user" if msg["role"] == "user" else "model"
            history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        try:
            chat = self._client.start_chat(history=history)
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=self._get_generation_config(kwargs),
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(f"Gemini stream chat API call failed: {str(e)}")
    
    def _get_generation_config(self, kwargs: Dict[str, Any]):
        """获取生成配置"""
        import google.generativeai as genai
        return genai.GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            import google.generativeai as genai
            models = genai.list_models()
            return [model.name.split('/')[-1] for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            raise RuntimeError(f"Failed to get available models: {str(e)}")