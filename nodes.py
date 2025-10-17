import torch
import numpy as np
from PIL import Image
import json
import io
import subprocess
import sys
import os
from svgpathtools import parse_path,Path
from xml.etree import ElementTree as ET
import cv2
import svgwrite

# 导入视频处理工具
from .video_utils import VideoMerger, get_available_transitions, get_supported_video_extensions

# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"

# 辅助函数

# 这些函数已经移到video_utils.py中

if sys.platform == "win32":
    executable_path = "potrace.exe" # Windows平台上的potrace可执行文件
else:
    executable_path = "potrace" # 非Windows平台上的potrace可执行文件

def sketch2svg(
    image: Image.Image,
    preserve_color: bool = False
        ) -> str:
    
    if preserve_color:
        # 转为RGBA，确保透明度处理
        rgba = image.convert("RGBA")
        w, h = rgba.size

        # 初始化SVG（使用显式px单位，避免渲染器单位换算导致尺寸异常）
        dwg = svgwrite.Drawing(size=(f"{w}px", f"{h}px"), profile="tiny", viewBox=f"0 0 {w} {h}")
        dwg.attribs["shape-rendering"] = "geometricPrecision"

        # 转为BGR用于KMeans聚类
        rgb = np.array(rgba)[..., :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # KMeans色彩量化
        labels, centers = kmeans_color_quantization(bgr, k=16)

        # 遍历每种颜色，单独提取mask → potrace → 添加到SVG
        for idx, color in enumerate(centers):
            mask = (labels == idx).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue

            # 将 mask 转换为 PBM（二进制 P4）字节，potrace 需要 PBM 输入
            pbm_data = mask_to_pbm_bytes(mask)

            # 调用potrace得到路径
            proc = subprocess.run(
                [f"{executable_path}", "-s", "--group", "-o", "-"],
                input=pbm_data,
                stdout=subprocess.PIPE,
                check=True
            )
            svg_layer = ET.fromstring(proc.stdout)

            # 继承potrace生成的坐标变换（通常包含y轴翻转和位移），否则路径会错位或比例异常
            NS = "{http://www.w3.org/2000/svg}"
            group_transform = None
            # 常见结构为 <svg><g transform="..."> <path .../> ... </g></svg>
            g_elem = svg_layer.find(f".//{NS}g[@transform]")
            if g_elem is not None and g_elem.get("transform"):
                group_transform = g_elem.get("transform")

            # 把每个path加到dwg中，并设置原色
            hex_color = hex_from_bgr(color)
            for path in svg_layer.findall(f".//{NS}path"):
                d = path.get("d")
                new_path = dwg.path(d=d, fill=hex_color, stroke="none")
                if group_transform:
                    new_path.update({"transform": group_transform})
                dwg.add(new_path)

        return dwg.tostring()

    else:
        # 黑白逻辑保持不变
        gray = image.convert("L")
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
        return svg_data.decode('utf-8')

def split_svg_by_subpaths(svg_bytes: bytes) -> bytes:
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

            outers = [(s,sp) for s,sp in parsed if not is_hole(sp)]
            holes  = [(s,sp) for s,sp in parsed if     is_hole(sp)]

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
                    ne.set('fill-rule', 'evenodd')
                parent.insert(idx+i, ne)

    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding='utf-8', xml_declaration=True)
    return buf.getvalue()

def is_hole(subpath: Path) -> bool:
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
    inside = False
    n = len(poly)
    for i in range(n):
        x0,y0 = poly[i]
        x1,y1 = poly[(i+1)%n]
        if ((y0>y) != (y1>y)) and (x < (x1-x0)*(y-y0)/(y1-y0) + x0):
            inside = not inside
    return inside

def hole_belongs_to_outer(hole: Path, outer: Path) -> bool:
    hpts = [seg.start for seg in hole]
    if not hpts:
        return False
    cx = sum(p.real for p in hpts) / len(hpts)
    cy = sum(p.imag for p in hpts) / len(hpts)
    opts = [seg.start for seg in outer]
    opts.append(outer[-1].end)
    poly = [(p.real,p.imag) for p in opts]
    return point_in_poly(cx, cy, poly)

def kmeans_color_quantization(bgr: np.ndarray, k: int, attempts: int = 3):
    Z = bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(bgr.shape[:2])
    centers = centers.astype(np.uint8)
    return labels, centers

def mask_to_pbm_bytes(mask: np.ndarray) -> bytes:
    """
    Convert a 2D uint8 mask (0 or 255) to binary PBM (P4) bytes for potrace.
    Foreground pixels should be 1 (black) and background 0 (white).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    bits = (mask != 0).astype(np.uint8)
    row_pad = (8 - (w % 8)) % 8
    packed = bytearray()
    for y in range(h):
        row = bits[y]
        if row_pad:
            row = np.concatenate([row, np.zeros(row_pad, dtype=np.uint8)])
        byte = 0
        for i, bit in enumerate(row):
            byte = (byte << 1) | int(bit)
            if (i % 8) == 7:
                packed.append(byte)
                byte = 0
    header = f"P4\n{w} {h}\n".encode("ascii")
    return header + bytes(packed)

def hex_from_bgr(bgr: np.ndarray) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"

class MultiImageMerger:
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
        if isinstance(images, list):
            if not all(isinstance(img, torch.Tensor) for img in images):
                raise TypeError("If images is a list, all its elements must be torch.Tensor.")
            image_tensors = images
        elif isinstance(images, torch.Tensor):
            if images.dim() == 4:
                image_tensors = [images[i] for i in range(images.shape[0])]
            elif images.dim() == 3:
                image_tensors = [images]
            else:
                raise ValueError(f"不支持的图片张量维度: {images.dim()}")
        else:
            raise TypeError(f"不支持的图片输入类型: {type(images)}")

        if len(image_tensors) == 0:
            raise ValueError("图片列表不能为空")
        
        pil_images = []
        for img_tensor in image_tensors:
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)
        
        total_images = len(pil_images)
        if direction == "horizontal":
            cols = items_per_row
            rows = (total_images + cols - 1) // cols
        else:
            rows = items_per_row
            cols = (total_images + rows - 1) // rows
        
        if direction == "horizontal":
            target_height = min(img.height for img in pil_images)
            resized_images = []
            for img in pil_images:
                aspect_ratio = img.width / img.height
                new_width = int(target_height * aspect_ratio)
                resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
        else:
            target_width = min(img.width for img in pil_images)
            resized_images = []
            for img in pil_images:
                aspect_ratio = img.height / img.width
                new_height = int(target_width * aspect_ratio)
                resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
        
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
        
        canvas_width = sum(col_widths)
        canvas_height = sum(row_heights)
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        y_offset = 0
        for row in range(rows):
            x_offset = 0
            for col in range(cols):
                idx = row * cols + col
                if idx < len(resized_images):
                    img = resized_images[idx]
                    x_pos = x_offset + (col_widths[col] - img.width) // 2
                    y_pos = y_offset + (row_heights[row] - img.height) // 2
                    canvas.paste(img, (x_pos, y_pos))
                x_offset += col_widths[col]
            y_offset += row_heights[row]
        
        canvas_array = np.array(canvas).astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_array).unsqueeze(0)
        return (canvas_tensor,)

class JSONExtractor:
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
        try:
            data = json.loads(json_text)
            key_parts = extract_key.split('.')
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preserve_color": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SVG_DATA",)
    FUNCTION = "convert_image_to_svg"
    CATEGORY = "WYQilin/image"

    def convert_image_to_svg(self, image: torch.Tensor, preserve_color: bool):
        if image.dim() == 4:
            image = image[0]
        
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        svg_data = sketch2svg(img, preserve_color)
        return (svg_data,)

class ImageDuplicator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "copy_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "duplicate_image"
    CATEGORY = "WYQilin/image"
    
    def duplicate_image(self, image, copy_count):
        # 检查输入是否为有效的图像张量
        if not isinstance(image, torch.Tensor):
            raise TypeError("输入必须是图像张量")
        
        # 确保图像是4D张量 [batch_size, height, width, channels]
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加batch维度
        elif image.dim() != 4:
            raise ValueError(f"不支持的图像维度: {image.dim()}")
        
        # 复制图像
        result_images = []
        for _ in range(copy_count):
            result_images.append(image.clone())
        
        # 将列表中的图像沿着batch维度连接
        result = torch.cat(result_images, dim=0)
        
        return (result,)

class StringLineBreaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "max_chars": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "add_line_breaks"
    CATEGORY = "WYQilin/text"
    
    def add_line_breaks(self, text, max_chars):
        if not text:
            return ("",)
        
        result = []
        current_line = ""
        
        for char in text:
            current_line += char
            if len(current_line) >= max_chars:
                result.append(current_line)
                current_line = ""
        
        if current_line:
            result.append(current_line)
        
        return ("\n".join(result),)

class VideoMergerWithTransitions:
    """视频合并转场节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "video_paths": ("STRING", {"multiline": True, "default":"video1.mp4\nvideo2.mp4", "tooltip": "视频文件路径列表，每行一个路径（List of video file paths, one path per line）"}),
                "reference_video_index": ("INT", {"default":0, "min":0, "step":1, "tooltip": "参考视频的索引（从0开始），决定输出视频的尺寸（Index of reference video (starting from 0), determines the size of output video）"}),
                "target_fps": ("FLOAT", {"default":30.0, "min":1.0, "max":60.0, "step":1.0, "display":"number", "tooltip": "目标帧率，所有视频将转换为该帧率并保持总时长不变（Target frame rate, all videos will be converted to this frame rate while maintaining total duration）"}),
                "device": (["cpu","cuda"], {"default":device,}),
                "transition": (get_available_transitions(),{"default": "fade", "tooltip": "转场效果类型（Transition effect type）"}),
                "transition_duration": ("FLOAT",{"default":1,"min":0.1,"max":3.0,"step":0.1,"display":"number","tooltip": "转场持续时间，单位秒，最大值为3秒（Transition duration in seconds, maximum 3 seconds）"}),
                "output_path": ("STRING", {"default":"merged_videos", "tooltip": "相对于ComfyUI的output目录的输出路径（Output path relative to ComfyUI's output directory）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_video_path",)
    FUNCTION = "merge_videos"
    OUTPUT_NODE = True
    CATEGORY = "WYQilin/video"
  
    def merge_videos(self, video_paths, reference_video_index, target_fps, device, transition, transition_duration, output_path):
        """合并视频并添加转场效果
        
        Args:
            video_paths: 视频文件路径列表（多行字符串）
            reference_video_index: 参考视频索引
            target_fps: 目标帧率
            device: 处理设备（cpu或cuda）
            transition: 转场效果类型
            transition_duration: 转场持续时间（秒）
            output_path: 相对于ComfyUI的output目录的输出路径
            
        Returns:
            str: 合并后的视频路径
        """
        try:
            # 处理视频路径列表
            paths = [path.strip() for path in video_paths.strip().split('\n') if path.strip()]
            
            # 创建VideoMerger实例
            merger = VideoMerger()
            
            # 获取ComfyUI的output目录（假设当前目录是自定义节点目录，parent的parent是ComfyUI根目录）
            comfyui_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_full_path = os.path.join(comfyui_root, "output", output_path.strip())
            
            # 调用合并功能
            merged_video_path = merger.merge_videos_with_transitions(
                video_paths=paths,
                reference_video_index=reference_video_index,
                target_fps=target_fps,
                transition_type=transition,
                transition_duration=transition_duration,
                output_path=output_full_path,
                device=device
            )
            
            return (merged_video_path,)
            
        except Exception as e:
            print(f"视频合并出现问题：{str(e)}")
            raise ValueError(f"视频合并失败：{str(e)}")
