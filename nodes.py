import torch
import numpy as np
from PIL import Image, ImageOps
import json
import io
import os
import subprocess
import sys
import tempfile
from svgpathtools import parse_path,Path
from xml.etree import ElementTree as ET

if sys.platform == "win32":
    executable_path = "potrace.exe" # Windows平台上的potrace可执行文件
else:
    executable_path = "potrace" # 非Windows平台上的potrace可执行文件

def sketch2svg(
    img_path: str,
    output_path: str =""
        ):
    
    print(f"Processing image: {img_path}", file=sys.stderr)
    im = Image.open(img_path)
    gray = im.convert("L")
    bw = gray.point(lambda x: 0 if x < 128 else 1, mode='1')
    buf = io.BytesIO()
    bw.save(buf, format='BMP')
    pbm_data = buf.getvalue()
    
    proc = subprocess.run(
            [f"{executable_path}", "-s", "--group", "-o", "-"],
            input=pbm_data,
            stdout=subprocess.PIPE,
            check=True
    )

    svg_data = split_svg_by_subpaths(proc.stdout) # 使用 split_svg_by_subpaths 处理SVG数据

    if not output_path:
        output_path = img_path.rsplit('.', 1)[0] + ".svg"
    with open(output_path, "wb") as f:
        f.write(svg_data)
    
    print(f"SVG saved to: {output_path}", file=sys.stderr)
    
    return output_path

def split_svg_by_subpaths(svg_bytes: bytes) -> bytes:
    """
    1. 分解SVG <path> 为子路径,
    2. 检查每个子路径是孔洞还是外部路径,
    3. 将外部路径与它们的孔洞分组,
    4. 为孔洞生成带有 fill-rule="evenodd" 的新 <path> 元素.
    5. 返回修改后的SVG字节数据。
    """
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_bytes)
    NS = "{http://www.w3.org/2000/svg}"
    style_attrs = ['fill','stroke','stroke-width','fill-rule','style']

    for parent in root.findall(".//"):
        for child in list(parent):
            if child.tag != NS + 'path':
                continue

            d_abs = parse_path(child.get('d')).d()

            parts = ['M'+p for p in d_abs.strip().split('M') if p.strip()]
            parsed = [(p, parse_path(p)) for p in parts]

            outers = [(s,sp) for s,sp in parsed if not is_hole(sp)] # 外部路径
            holes  = [(s,sp) for s,sp in parsed if     is_hole(sp)] # 孔洞路径

            used_holes = set()
            groups = []
            for o_str,o_path in outers:
                grp_holes = []
                for h_str,h_path in holes:
                    if h_str in used_holes:
                        continue
                    if hole_belongs_to_outer(h_path, o_path):
                        grp_holes.append(h_str)
                        used_holes.add(h_str)
                groups.append((o_str, grp_holes))

            for h_str,h_path in holes:
                if h_str not in used_holes:
                    groups.append((h_str, []))

            own_attrs = dict(child.attrib)
            inherited = {} 
            if parent.tag == NS+'g':
                for attr in style_attrs:
                    if attr not in own_attrs and parent.get(attr) is not None:
                        inherited[attr] = parent.get(attr)

            idx = list(parent).index(child)
            parent.remove(child)
            for i,(outer_str, hole_strs) in enumerate(groups):
                ne = ET.Element(NS+'path')
                for k,v in inherited.items():
                    ne.set(k,v)
                for k,v in own_attrs.items():
                    if k!='d':
                        ne.set(k,v)
                combined = outer_str + ''.join(hole_strs)
                ne.set('d', combined)
                if hole_strs:
                    ne.set('fill-rule', 'evenodd') # 设置填充规则为 "evenodd" 以正确显示孔洞
                parent.insert(idx+i, ne)

    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding='utf-8', xml_declaration=True)
    return buf.getvalue()

def is_hole(subpath: Path) -> bool:
    """
    通过检查子路径片段的有符号面积来判断其是否为“孔洞”。
    如果面积为负，则为孔洞（逆时针方向）。
    """
    pts = [seg.start for seg in subpath]
    if not pts:
        return False
    pts.append(subpath[-1].end)
    area = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i].real, pts[i].imag
        x1, y1 = pts[i+1].real, pts[i+1].imag
        area += x0*y1 - x1*y0
    area *= 0.5
    return area < 0

def point_in_poly(x: float, y: float, poly: list[tuple[float,float]]) -> bool:
    """
    使用射线投射算法判断点(x,y)是否在由顶点列表定义的多边形内部。
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x0,y0 = poly[i]
        x1,y1 = poly[(i+1)%n]
        if ((y0>y) != (y1>y)) and (x < (x1-x0)*(y-y0)/(y1-y0) + x0):
            inside = not inside
    return inside

def hole_belongs_to_outer(hole: Path, outer: Path) -> bool:
    """
    通过检查孔洞点的质心是否在外部路径多边形内部，判断孔洞是否属于外部路径。
    """
    hpts = [seg.start for seg in hole]
    if not hpts:
        return False
    cx = sum(p.real for p in hpts) / len(hpts)
    cy = sum(p.imag for p in hpts) / len(hpts)
    opts = [seg.start for seg in outer]
    opts.append(outer[-1].end)
    poly = [(p.real,p.imag) for p in opts]
    return point_in_poly(cx, cy, poly)

class MultiImageMerger:
    """
    多图合并节点 - 将任意大小的图片列表合并到一张图上
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}), # 拼接方向
                "items_per_row": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}), # 每行/列的图片数量
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
            if images.dim() == 4:  # 批处理张量 [B, H, W, C]
                image_tensors = [images[i] for i in range(images.shape[0])]
            elif images.dim() == 3:  # 单张图片张量 [H, W, C]
                image_tensors = [images]
            else:
                raise ValueError(f"不支持的图片张量维度: {images.dim()}")
        else:
            raise TypeError(f"不支持的图片输入类型: {type(images)}")

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
        else:  # vertical 垂直
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
                "json_text": ("STRING", {"multiline": True, "default": "{}"}), # JSON格式的文本
                "extract_key": ("STRING", {"default": "key", "multiline": False}), # 要提取的key
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
                        raise KeyError(f"键 '{key_part}' 未找到")
                elif isinstance(current_data, list):
                    try:
                        index = int(key_part)
                        if 0 <= index < len(current_data):
                            current_data = current_data[index]
                        else:
                            raise IndexError(f"索引 {index} 超出范围")
                    except ValueError:
                        raise ValueError(f"无效的列表索引: '{key_part}'")
                else:
                    raise TypeError(f"无法在类型 {type(current_data)} 上访问键 '{key_part}'")
            
            # 将结果转换为字符串
            if isinstance(current_data, (dict, list)):
                result = json.dumps(current_data, ensure_ascii=False, indent=2)
            else:
                result = str(current_data)
            
            return (result,)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的JSON格式: {str(e)}")
        except (KeyError, IndexError, TypeError, ValueError):
            return ("None",)


class ImageToSVG:
    """
    图片转SVG节点 - 将输入的图片转换为SVG格式。
    依赖于potrace命令行工具，请确保已安装并配置好环境变量。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), # 输入图片
                "output_filename_prefix": ("STRING", {"default": "ComfyUI_WYQilin"}), # 输出SVG文件的前缀名
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}), # 输出SVG文件的路径，如果为空则保存到系统临时目录
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("SVG_DATA", "SVG_PATH",)
    FUNCTION = "convert_image_to_svg"
    CATEGORY = "WYQilin/image"

    def convert_image_to_svg(self, image: torch.Tensor, output_filename_prefix: str, output_path: str):
        if image.dim() == 4: # 批处理张量
            image = image[0] # 取批处理中的第一张图片
        
        # 将PyTorch张量转换为PIL图像
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 为输入图片创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input_img:
            temp_input_img_path = temp_input_img.name
            img.save(temp_input_img_path)
        
        # 确定输出路径
        if not output_path:
            output_dir = tempfile.gettempdir() # 如果未指定，则使用系统临时目录
        else:
            output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)
            
        # 构造输出文件路径
        output_file_name = f"{output_filename_prefix}.svg"
        output_svg_path = os.path.join(output_dir, output_file_name)
        
        # 将图片转换为SVG
        final_output_path = sketch2svg(temp_input_img_path, output_svg_path)
        
        # 读取SVG数据
        with open(final_output_path, "rb") as f:
            svg_data = f.read().decode('utf-8')
        
        # 清理临时输入图片文件
        os.remove(temp_input_img_path)
        
        return (svg_data, final_output_path,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MultiImageMerger": MultiImageMerger,
    "JSONExtractor": JSONExtractor,
    "ImageToSVG": ImageToSVG,
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiImageMerger": "多图合并",
    "JSONExtractor": "JSON提取器",
    "ImageToSVG": "图片转SVG",
}
