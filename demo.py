#!/usr/bin/env python3
"""
AI模型适配器演示脚本
"""

from ai_model_adapter import ModelManager, ModelConfig, ModelType

def demo_basic_usage():
    """演示基本使用方法"""
    print("=== AI模型适配器演示 ===\n")
    
    # 1. 创建模型管理器
    print("1. 创建模型管理器...")
    manager = ModelManager()
    print("   ✓ 模型管理器创建成功")
    
    # 2. 查看可用模型
    print("\n2. 查看可用模型:")
    models = manager.list_available_models()
    for model in models:
        print(f"   - {model}")
    
    # 3. 添加自定义模型
    print("\n3. 添加自定义模型...")
    custom_config = ModelConfig(
        model_type=ModelType.OPENAI,
        model_name="gpt-4o-mini",
        api_key="your-api-key-here",  # 在实际使用中请设置真实的API密钥
        temperature=0.7,
        max_tokens=1000
    )
    manager.add_model("custom-gpt", custom_config)
    print("   ✓ 自定义模型添加成功")
    
    # 4. 获取模型信息
    print("\n4. 获取模型信息:")
    try:
        info = manager.get_model_info("custom-gpt")
        print(f"   - 模型类型: {info['model_type']}")
        print(f"   - 模型名称: {info['model_name']}")
        print(f"   - 最大令牌: {info['max_tokens']}")
        print(f"   - 温度: {info['temperature']}")
    except Exception as e:
        print(f"   ✗ 获取模型信息失败: {e}")
    
    # 5. 模拟API调用（需要真实的API密钥）
    print("\n5. 模拟API调用:")
    print("   注意：需要设置真实的API密钥才能实际调用模型")
    print("   示例调用代码：")
    print("   ```python")
    print("   response = manager.generate('Hello, world!', model_name='custom-gpt')")
    print("   print(response.content)")
    print("   ```")
    
    # 6. 设置默认模型
    print("\n6. 设置默认模型...")
    try:
        manager.set_default_model("custom-gpt")
        print("   ✓ 默认模型设置成功")
    except Exception as e:
        print(f"   ✗ 设置默认模型失败: {e}")
    
    print("\n=== 演示完成 ===")

def demo_advanced_features():
    """演示高级功能"""
    print("\n=== 高级功能演示 ===\n")
    
    manager = ModelManager()
    
    # 1. 模型比较功能
    print("1. 模型比较功能:")
    print("   可以同时调用多个模型并比较结果")
    print("   示例代码：")
    print("   ```python")
    print("   results = manager.compare_models(")
    print("       '解释人工智能',")
    print("       ['gpt-3.5-turbo', 'claude-3-sonnet', 'gemini-pro']")
    print("   )")
    print("   for model, response in results.items():")
    print("       print(f'{model}: {response.content}')")
    print("   ```")
    
    # 2. 批量生成功能
    print("\n2. 批量生成功能:")
    print("   可以批量处理多个提示")
    print("   示例代码：")
    print("   ```python")
    print("   prompts = ['什么是AI?', '什么是ML?', '什么是DL?']")
    print("   results = manager.batch_generate(prompts)")
    print("   for i, result in enumerate(results):")
    print("       print(f'问题{i+1}: {result.content}')")
    print("   ```")
    
    # 3. 流式生成功能
    print("\n3. 流式生成功能:")
    print("   支持实时流式输出")
    print("   示例代码：")
    print("   ```python")
    print("   for chunk in manager.stream_generate('写一首诗'):")
    print("       print(chunk, end='')")
    print("   ```")
    
    # 4. 健康检查功能
    print("\n4. 健康检查功能:")
    print("   可以检查模型是否可用")
    print("   示例代码：")
    print("   ```python")
    print("   health = manager.health_check('gpt-3.5-turbo')")
    print("   print(f'状态: {health[\"status\"]}')")
    print("   ```")
    
    print("\n=== 高级功能演示完成 ===")

def demo_configuration():
    """演示配置管理"""
    print("\n=== 配置管理演示 ===\n")
    
    print("1. 配置文件位置:")
    print("   默认配置文件: ai_model_config.yaml")
    print("   可以通过环境变量或代码指定自定义配置文件")
    
    print("\n2. 环境变量配置:")
    print("   OPENAI_API_KEY=your_openai_key")
    print("   ANTHROPIC_API_KEY=your_anthropic_key")
    print("   GOOGLE_API_KEY=your_google_key")
    
    print("\n3. 配置文件示例:")
    print("   ```yaml")
    print("   default_model: gpt-3.5-turbo")
    print("   models:")
    print("     gpt-3.5-turbo:")
    print("       model_type: openai")
    print("       model_name: gpt-3.5-turbo")
    print("       api_key_env: OPENAI_API_KEY")
    print("       temperature: 0.7")
    print("       max_tokens: 4000")
    print("   ```")
    
    print("\n=== 配置管理演示完成 ===")

def main():
    """主函数"""
    print("🤖 AI模型适配器演示程序")
    print("=" * 50)
    
    demo_basic_usage()
    demo_advanced_features()
    demo_configuration()
    
    print("\n" + "=" * 50)
    print("📖 更多信息请查看 README.md")
    print("🧪 运行测试: python3 test_adapter.py")
    print("📝 完整示例: python3 example.py")
    print("=" * 50)

if __name__ == "__main__":
    main()