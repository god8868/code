"""
配置管理器
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from .base import ModelType, ModelConfig


class Config:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "ai_model_config.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        self._config = yaml.safe_load(f) or {}
                    else:
                        self._config = json.load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config file {self.config_file}: {e}")
                self._config = {}
        else:
            self._config = {}
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save config file: {e}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model_name not in self._config.get("models", {}):
            return None
        
        model_conf = self._config["models"][model_name]
        return ModelConfig(
            model_type=ModelType(model_conf["model_type"]),
            model_name=model_conf["model_name"],
            api_key=model_conf.get("api_key") or os.getenv(model_conf.get("api_key_env", "")),
            base_url=model_conf.get("base_url"),
            max_tokens=model_conf.get("max_tokens", 4000),
            temperature=model_conf.get("temperature", 0.7),
            timeout=model_conf.get("timeout", 30),
            extra_params=model_conf.get("extra_params", {})
        )
    
    def add_model_config(self, alias: str, config: ModelConfig):
        """添加模型配置"""
        if "models" not in self._config:
            self._config["models"] = {}
        
        self._config["models"][alias] = {
            "model_type": config.model_type.value,
            "model_name": config.model_name,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "extra_params": config.extra_params
        }
        self.save_config()
    
    def remove_model_config(self, alias: str):
        """删除模型配置"""
        if "models" in self._config and alias in self._config["models"]:
            del self._config["models"][alias]
            self.save_config()
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """列出所有模型配置"""
        return self._config.get("models", {})
    
    def get_default_model(self) -> Optional[str]:
        """获取默认模型"""
        return self._config.get("default_model")
    
    def set_default_model(self, model_name: str):
        """设置默认模型"""
        self._config["default_model"] = model_name
        self.save_config()
    
    def create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "default_model": "gpt-3.5-turbo",
            "models": {
                "gpt-3.5-turbo": {
                    "model_type": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "api_key_env": "OPENAI_API_KEY",
                    "max_tokens": 4000,
                    "temperature": 0.7,
                    "timeout": 30,
                    "extra_params": {}
                },
                "gpt-4": {
                    "model_type": "openai",
                    "model_name": "gpt-4",
                    "api_key_env": "OPENAI_API_KEY",
                    "max_tokens": 4000,
                    "temperature": 0.7,
                    "timeout": 30,
                    "extra_params": {}
                },
                "claude-3-sonnet": {
                    "model_type": "claude",
                    "model_name": "claude-3-sonnet-20240229",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "max_tokens": 4000,
                    "temperature": 0.7,
                    "timeout": 30,
                    "extra_params": {}
                },
                "gemini-pro": {
                    "model_type": "gemini",
                    "model_name": "gemini-pro",
                    "api_key_env": "GOOGLE_API_KEY",
                    "max_tokens": 4000,
                    "temperature": 0.7,
                    "timeout": 30,
                    "extra_params": {}
                }
            }
        }
        
        self._config = default_config
        self.save_config()
        return default_config