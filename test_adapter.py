"""
AI模型适配器测试文件
"""

import unittest
from unittest.mock import Mock, patch
from ai_model_adapter import ModelManager, ModelConfig, ModelType, Config


class TestModelAdapter(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        self.config.create_default_config()
        self.manager = ModelManager(self.config)
    
    def test_model_manager_initialization(self):
        """测试模型管理器初始化"""
        self.assertIsInstance(self.manager, ModelManager)
        self.assertIsInstance(self.manager.config, Config)
    
    def test_add_model_config(self):
        """测试添加模型配置"""
        test_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="test-model",
            api_key="test-key",
            temperature=0.5,
            max_tokens=1000
        )
        
        self.manager.add_model("test-model", test_config)
        models = self.manager.list_available_models()
        self.assertIn("test-model", models)
    
    def test_remove_model_config(self):
        """测试删除模型配置"""
        # 先添加一个模型
        test_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="test-model",
            api_key="test-key"
        )
        self.manager.add_model("test-model", test_config)
        
        # 然后删除
        self.manager.remove_model("test-model")
        models = self.manager.list_available_models()
        self.assertNotIn("test-model", models)
    
    def test_set_default_model(self):
        """测试设置默认模型"""
        # 添加一个测试模型
        test_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="test-model",
            api_key="test-key"
        )
        self.manager.add_model("test-model", test_config)
        
        # 设置为默认模型
        self.manager.set_default_model("test-model")
        default_model = self.manager.config.get_default_model()
        self.assertEqual(default_model, "test-model")
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的模型类型
        with self.assertRaises(ValueError):
            ModelConfig(
                model_type="invalid_type",
                model_name="test-model"
            )
    
    def test_model_info(self):
        """测试获取模型信息"""
        # 需要有效的API密钥才能测试实际的模型信息
        # 这里只测试配置存在的情况
        models = self.manager.list_available_models()
        self.assertGreater(len(models), 0)
    
    def test_health_check_no_api_key(self):
        """测试在没有API密钥时的健康检查"""
        # 这个测试会失败，因为没有真实的API密钥
        health = self.manager.health_check("gpt-3.5-turbo")
        self.assertEqual(health['status'], 'error')
        self.assertIn('error', health)
    
    def test_batch_generate_error_handling(self):
        """测试批量生成的错误处理"""
        prompts = ["Test prompt 1", "Test prompt 2"]
        
        # 这会因为没有API密钥而失败，但应该返回错误响应
        results = self.manager.batch_generate(prompts, model_name="gpt-3.5-turbo")
        
        self.assertEqual(len(results), len(prompts))
        for result in results:
            self.assertTrue(result.metadata.get('error', False))


class TestConfig(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config("test_config.yaml")
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        default_config = self.config.create_default_config()
        
        self.assertIn("default_model", default_config)
        self.assertIn("models", default_config)
        self.assertIn("gpt-3.5-turbo", default_config["models"])
    
    def test_model_config_crud(self):
        """测试模型配置的增删改查"""
        # 创建测试配置
        test_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="test-model",
            api_key="test-key",
            temperature=0.8,
            max_tokens=2000
        )
        
        # 添加配置
        self.config.add_model_config("test-model", test_config)
        
        # 读取配置
        retrieved_config = self.config.get_model_config("test-model")
        self.assertIsNotNone(retrieved_config)
        self.assertEqual(retrieved_config.model_name, "test-model")
        self.assertEqual(retrieved_config.temperature, 0.8)
        
        # 删除配置
        self.config.remove_model_config("test-model")
        retrieved_config = self.config.get_model_config("test-model")
        self.assertIsNone(retrieved_config)
    
    def tearDown(self):
        """清理测试环境"""
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")


def run_simple_test():
    """运行简单的功能测试"""
    print("=== 运行简单功能测试 ===\n")
    
    try:
        # 测试配置创建
        config = Config("test_simple.yaml")
        config.create_default_config()
        print("✓ 配置文件创建成功")
        
        # 测试模型管理器
        manager = ModelManager(config)
        print("✓ 模型管理器创建成功")
        
        # 测试模型列表
        models = manager.list_available_models()
        print(f"✓ 可用模型: {models}")
        
        # 测试添加模型
        test_config = ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="test-model",
            api_key="test-key"
        )
        manager.add_model("test-model", test_config)
        print("✓ 模型添加成功")
        
        # 测试模型信息
        info = manager.get_model_info("test-model")
        print(f"✓ 模型信息: {info}")
        
        # 测试健康检查（预期失败）
        health = manager.health_check("test-model")
        print(f"✓ 健康检查: {health['status']}")
        
        print("\n=== 所有基本功能测试通过 ===")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
    
    finally:
        # 清理测试文件
        import os
        if os.path.exists("test_simple.yaml"):
            os.remove("test_simple.yaml")


if __name__ == "__main__":
    print("AI模型适配器测试\n")
    
    # 运行简单测试
    run_simple_test()
    
    print("\n" + "="*50)
    print("运行单元测试:")
    print("="*50)
    
    # 运行单元测试
    unittest.main(argv=[''], exit=False, verbosity=2)