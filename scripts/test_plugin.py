"""
测试脚本 - 验证插件功能
"""
import sys
import os
import json
import torch
import numpy as np
from PIL import Image

# 添加插件路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nodes import MultiImageMerger, JSONExtractor

def test_multi_image_merger():
    """测试多图合并节点"""
    print("测试多图合并节点...")
    
    # 创建测试图片
    test_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    sizes = [(100, 150), (120, 100), (80, 200), (150, 120)]
    
    for i, (color, size) in enumerate(zip(colors, sizes)):
        # 创建纯色图片
        img_array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)
        test_images.append(img_tensor)
    
    # 测试横向拼接
    merger = MultiImageMerger()
    result_h = merger.merge_images(test_images, "horizontal", 2)
    print(f"横向拼接结果尺寸: {result_h[0].shape}")
    
    # 测试竖向拼接
    result_v = merger.merge_images(test_images, "vertical", 2)
    print(f"竖向拼接结果尺寸: {result_v[0].shape}")
    
    print("多图合并节点测试完成！")

def test_json_extractor():
    """测试JSON提取器节点"""
    print("\n测试JSON提取器节点...")
    
    # 创建测试JSON数据
    test_json = {
        "users": [
            {
                "id": 1,
                "name": "张三",
                "profile": {
                    "age": 25,
                    "city": "北京"
                }
            },
            {
                "id": 2,
                "name": "李四",
                "profile": {
                    "age": 30,
                    "city": "上海"
                }
            }
        ],
        "total": 2,
        "status": "success"
    }
    
    json_text = json.dumps(test_json, ensure_ascii=False)
    extractor = JSONExtractor()
    
    # 测试各种提取场景
    test_cases = [
        ("status", "success"),
        ("total", "2"),
        ("users.0.name", "张三"),
        ("users.1.profile.city", "上海"),
        ("users.0.profile", '{\n  "age": 25,\n  "city": "北京"\n}'),
    ]
    
    for key, expected in test_cases:
        try:
            result = extractor.extract_json(json_text, key)
            print(f"提取 '{key}': {result[0]}")
            # 简单验证（实际使用中可能需要更严格的比较）
            if expected in result[0]:
                print("✓ 测试通过")
            else:
                print("✗ 测试失败")
        except Exception as e:
            print(f"✗ 提取 '{key}' 失败: {e}")
    
    print("JSON提取器节点测试完成！")

if __name__ == "__main__":
    test_multi_image_merger()
    test_json_extractor()
    print("\n所有测试完成！")
