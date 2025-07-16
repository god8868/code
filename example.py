"""
AI模型适配器使用示例
"""

import os
from ai_model_adapter import ModelManager, ModelConfig, ModelType, Config

def main():
    # 创建配置管理器
    config = Config()
    
    # 如果没有配置文件，创建默认配置
    if not os.path.exists("ai_model_config.yaml"):
        config.create_default_config()
        print("创建了默认配置文件: ai_model_config.yaml")
    
    # 创建模型管理器
    manager = ModelManager(config)
    
    # 设置环境变量（实际使用时请设置真实的API Key）
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    # os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"  
    # os.environ["GOOGLE_API_KEY"] = "your_google_api_key"
    
    print("=== AI模型适配器示例 ===\n")
    
    # 1. 列出可用模型
    print("1. 可用模型:")
    models = manager.list_available_models()
    for model in models:
        print(f"   - {model}")
    print()
    
    # 2. 动态添加模型配置
    print("2. 动态添加模型配置:")
    new_config = ModelConfig(
        model_type=ModelType.OPENAI,
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY", "your_api_key"),
        temperature=0.5,
        max_tokens=2000
    )
    manager.add_model("gpt-4o", new_config)
    print("   添加了 gpt-4o 模型配置")
    print()
    
    # 3. 单次生成示例
    print("3. 单次生成示例:")
    try:
        response = manager.generate(
            "请用中文介绍一下人工智能",
            model_name="gpt-3.5-turbo",
            max_tokens=100
        )
        print(f"   模型: {response.model_name}")
        print(f"   响应: {response.content}")
        print(f"   用量: {response.usage}")
        print(f"   延迟: {response.latency:.2f}秒")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 4. 对话模式示例
    print("4. 对话模式示例:")
    try:
        messages = [
            {"role": "user", "content": "你好！"},
            {"role": "assistant", "content": "你好！有什么我可以帮助您的吗？"},
            {"role": "user", "content": "请介绍一下Python编程语言"}
        ]
        
        response = manager.chat(messages, model_name="gpt-3.5-turbo")
        print(f"   模型: {response.model_name}")
        print(f"   响应: {response.content}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 5. 流式生成示例
    print("5. 流式生成示例:")
    try:
        print("   生成中: ", end="")
        for chunk in manager.stream_generate(
            "请简单介绍一下机器学习",
            model_name="gpt-3.5-turbo",
            max_tokens=80
        ):
            print(chunk, end="")
        print()
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 6. 模型比较示例
    print("6. 模型比较示例:")
    try:
        prompt = "请用一句话解释什么是深度学习"
        comparison = manager.compare_models(
            prompt,
            ["gpt-3.5-turbo", "gpt-4"],
            max_tokens=50
        )
        
        for model_name, response in comparison.items():
            if isinstance(response, dict) and "error" in response:
                print(f"   {model_name}: 错误 - {response['error']}")
            else:
                print(f"   {model_name}: {response.content}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 7. 批量生成示例
    print("7. 批量生成示例:")
    try:
        prompts = [
            "什么是人工智能？",
            "什么是机器学习？",
            "什么是深度学习？"
        ]
        
        results = manager.batch_generate(prompts, model_name="gpt-3.5-turbo", max_tokens=30)
        for i, response in enumerate(results):
            print(f"   问题{i+1}: {response.content}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 8. 健康检查示例
    print("8. 健康检查示例:")
    health = manager.health_check("gpt-3.5-turbo")
    print(f"   状态: {health['status']}")
    if health['status'] == 'healthy':
        print(f"   延迟: {health['latency']:.2f}秒")
    else:
        print(f"   错误: {health.get('error', 'Unknown error')}")
    print()
    
    # 9. 获取模型信息
    print("9. 模型信息:")
    try:
        info = manager.get_model_info("gpt-3.5-turbo")
        print(f"   模型类型: {info['model_type']}")
        print(f"   模型名称: {info['model_name']}")
        print(f"   最大令牌: {info['max_tokens']}")
        print(f"   温度: {info['temperature']}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 10. 设置默认模型
    print("10. 设置默认模型:")
    try:
        manager.set_default_model("gpt-4o")
        print("   默认模型已设置为 gpt-4o")
        
        # 使用默认模型生成
        response = manager.generate("测试默认模型", max_tokens=20)
        print(f"   默认模型响应: {response.content}")
    except Exception as e:
        print(f"   错误: {e}")


def advanced_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===\n")
    
    # 创建自定义配置
    config = Config("custom_config.yaml")
    manager = ModelManager(config)
    
    # 添加多个模型配置
    configs = [
        ("fast-model", ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0.3,
            max_tokens=500
        )),
        ("creative-model", ModelConfig(
            model_type=ModelType.OPENAI,
            model_name="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0.9,
            max_tokens=1000
        ))
    ]
    
    for alias, config in configs:
        manager.add_model(alias, config)
    
    # 使用不同的模型完成不同的任务
    tasks = [
        ("fast-model", "请快速总结：机器学习是什么？"),
        ("creative-model", "请创作一首关于AI的诗歌")
    ]
    
    for model_name, prompt in tasks:
        try:
            response = manager.generate(prompt, model_name=model_name)
            print(f"模型 {model_name}:")
            print(f"任务: {prompt}")
            print(f"响应: {response.content}")
            print(f"用量: {response.usage}")
            print()
        except Exception as e:
            print(f"模型 {model_name} 错误: {e}")
            print()


if __name__ == "__main__":
    main()
    # advanced_example()  # 取消注释以运行高级示例