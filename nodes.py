import torch
import numpy as np
from PIL import Image, ImageOps
import json
import io

class MultiImageMerger:
    """
    多图合并节点 - 将任意大小的图片列表合并到一张图上
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "items_per_row": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "WYQilin/image"
    
    def merge_images(self, images, direction, items_per_row):
        """
        合并图片
        
        Args:
            images: 输入的图片张量列表
            direction: 拼接方向 ("horizontal" 或 "vertical")
            items_per_row: 每行/列的图片数量
        
        Returns:
            合并后的图片张量
        """
        if isinstance(images, list):
            if not all(isinstance(img, torch.Tensor) for img in images):
                raise TypeError("If images is a list, all its elements must be torch.Tensor.")
            image_tensors = images
        elif isinstance(images, torch.Tensor):
            if images.dim() == 4:  # Batch tensor [B, H, W, C]
                image_tensors = [images[i] for i in range(images.shape[0])]
            elif images.dim() == 3:  # Single image tensor [H, W, C]
                image_tensors = [images]
            else:
                raise ValueError(f"Unsupported image tensor dimension: {images.dim()}")
        else:
            raise TypeError(f"Unsupported images input type: {type(images)}")

        if len(image_tensors) == 0:
            raise ValueError("图片列表不能为空")
        
        # 将张量转换为PIL图片列表
        pil_images = []
        for img_tensor in image_tensors:
            
            # 假设输入张量格式为 [H, W, C] 且值在 [0, 1] 范围内
            
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)
        
        # 计算网格布局
        total_images = len(pil_images)
        if direction == "horizontal":
            cols = items_per_row
            rows = (total_images + cols - 1) // cols
        else:  # vertical
            rows = items_per_row
            cols = (total_images + rows - 1) // rows
        
        # 统一图片尺寸
        if direction == "horizontal":
            # 横向拼接：统一高度，保持宽度比例
            target_height = min(img.height for img in pil_images)
            resized_images = []
            for img in pil_images:
                aspect_ratio = img.width / img.height
                new_width = int(target_height * aspect_ratio)
                resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
        else:
            # 竖向拼接：统一宽度，保持高度比例
            target_width = min(img.width for img in pil_images)
            resized_images = []
            for img in pil_images:
                aspect_ratio = img.height / img.width
                new_height = int(target_width * aspect_ratio)
                resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
        
        # 计算每行/列的最大尺寸
        row_heights = []
        col_widths = []
        
        for row in range(rows):
            max_height = 0
            for col in range(cols):
                idx = row * cols + col
                if idx < len(resized_images):
                    max_height = max(max_height, resized_images[idx].height)
            row_heights.append(max_height)
        
        for col in range(cols):
            max_width = 0
            for row in range(rows):
                idx = row * cols + col
                if idx < len(resized_images):
                    max_width = max(max_width, resized_images[idx].width)
            col_widths.append(max_width)
        
        # 计算最终画布尺寸
        canvas_width = sum(col_widths)
        canvas_height = sum(row_heights)
        
        # 创建空白画布
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # 粘贴图片到画布
        y_offset = 0
        for row in range(rows):
            x_offset = 0
            for col in range(cols):
                idx = row * cols + col
                if idx < len(resized_images):
                    img = resized_images[idx]
                    # 居中对齐
                    x_pos = x_offset + (col_widths[col] - img.width) // 2
                    y_pos = y_offset + (row_heights[row] - img.height) // 2
                    canvas.paste(img, (x_pos, y_pos))
                x_offset += col_widths[col]
            y_offset += row_heights[row]
        
        # 转换回张量格式
        canvas_array = np.array(canvas).astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_array).unsqueeze(0)
        return (canvas_tensor,)


class JSONExtractor:
    """
    JSON提取器节点 - 从JSON文本中提取指定key的值
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": "{}"}),
                "extract_key": ("STRING", {"default": "key", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract_json"
    CATEGORY = "WYQilin/json"
    
    def extract_json(self, json_text, extract_key):
        """
        从JSON文本中提取指定key的值
        
        Args:
            json_text: JSON格式的文本
            extract_key: 要提取的key，支持多层级，用点号分隔
        
        Returns:
            提取到的值（转换为字符串）
        """
        try:
            # 解析JSON
            data = json.loads(json_text)
            
            # 分割key路径
            key_parts = extract_key.split('.')
            
            # 逐层提取
            current_data = data
            for key_part in key_parts:
                if isinstance(current_data, dict):
                    if key_part in current_data:
                        current_data = current_data[key_part]
                    else:
                        raise KeyError(f"Key '{key_part}' not found")
                elif isinstance(current_data, list):
                    try:
                        index = int(key_part)
                        if 0 <= index < len(current_data):
                            current_data = current_data[index]
                        else:
                            raise IndexError(f"Index {index} out of range")
                    except ValueError:
                        raise ValueError(f"Invalid list index: '{key_part}'")
                else:
                    raise TypeError(f"Cannot access key '{key_part}' on {type(current_data)}")
            
            # 将结果转换为字符串
            if isinstance(current_data, (dict, list)):
                result = json.dumps(current_data, ensure_ascii=False, indent=2)
            else:
                result = str(current_data)
            
            return (result,)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except (KeyError, IndexError, TypeError, ValueError):
            return ("None",)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MultiImageMerger": MultiImageMerger,
    "JSONExtractor": JSONExtractor,
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiImageMerger": "多图合并",
    "JSONExtractor": "JSON提取器",
}
