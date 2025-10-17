import os
import subprocess
import json
import tempfile

def get_supported_video_extensions():
    """返回支持的视频文件扩展名列表"""
    return ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

def get_video_info(video_path):
    """获取视频文件信息，包括时长、宽度、高度、帧率等"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return None
            
        stream = data['streams'][0]
        
        # 计算帧率
        fps_str = stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30
        else:
            fps = float(fps_str)
            
        return {
            'width': int(stream.get('width', 1920)),
            'height': int(stream.get('height', 1080)),
            'fps': fps,
            'duration': float(stream.get('duration', 0))
        }
    except Exception as e:
        print(f"获取视频信息错误: {e}")
        return None

def has_audio(video_path):
    """检查视频文件是否包含音频流"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        return len(data.get('streams', [])) > 0
    except Exception:
        return False

def get_audio_duration(video_path):
    """获取视频文件中音频的时长"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return 0
            
        stream = data['streams'][0]
        return float(stream.get('duration', 0))
    except Exception:
        return 0

def get_available_transitions():
    """返回FFmpeg支持的转场效果列表，包括xfade和gl-transition两大类"""
    return [
        # xfade转场类型
        'circleclose', 'circlecrop', 'circleopen', 'coverdown', 'coverleft',
        'coverright', 'coverup', 'diagbl', 'diagbr', 'diagtl',
        'diagtr', 'dissolve', 'fade', 'fadeblack', 'fadefast',
        'fadegrays', 'fadeslow', 'fadewhite', 'hlslice', 'hlwind',
        'horzclose', 'horzopen', 'hrslice', 'hrwind', 'pixelize',
        'radial', 'rectcrop', 'revealdown', 'revealleft', 'revealright',
        'revealup', 'slideleft', 'slideright', 'slidedown', 'slideup',
        'smoothdown', 'smoothleft', 'smoothright', 'smoothup', 'squeezeh',
        'squeezev', 'vdslice', 'vdwind', 'vertclose', 'vertopen',
        'vuslice', 'vuwind', 'wipebl', 'wipebr', 'wipedown',
        'wipeleft', 'wiperight', 'wipetl', 'wipetr', 'wipeup',
        'zoomin',
        
        # gl-transition转场类型
        'gl_angular', 'gl_Bars', 'gl_blend', 'gl_BookFlip', 'gl_Bounce',
        'gl_BowTie', 'gl_ButterflyWaveScrawler', 'gl_cannabisleaf', 'gl_chessboard',
        'gl_CornerVanish', 'gl_CrazyParametricFun', 'gl_crosshatch', 'gl_CrossOut',
        'gl_crosswarp', 'gl_CrossZoom', 'gl_cube', 'gl_Diamond', 'gl_DirectionalScaled',
        'gl_directionalwarp', 'gl_doorway', 'gl_DoubleDiamond', 'gl_Dreamy',
        'gl_EdgeTransition', 'gl_Exponential_Swish', 'gl_fadecolor', 'gl_FanIn',
        'gl_FanOut', 'gl_FanUp', 'gl_Flower', 'gl_GridFlip', 'gl_heart',
        'gl_hexagonalize', 'gl_InvertedPageCurl', 'gl_kaleidoscope', 'gl_LinearBlur',
        'gl_Lissajous_Tiles', 'gl_morph', 'gl_Mosaic', 'gl_perlin', 'gl_pinwheel',
        'gl_polar_function', 'gl_PolkaDotsCurtain', 'gl_powerKaleido', 'gl_randomNoisex',
        'gl_randomsquares', 'gl_ripple', 'gl_Rolls', 'gl_rotate_scale_fade',
        'gl_rotateTransition', 'gl_RotateScaleVanish', 'gl_SimpleBookCurl',
        'gl_SimplePageCurl', 'gl_Slides', 'gl_squareswire', 'gl_StageCurtains',
        'gl_StarWipe', 'gl_static_wipe', 'gl_StereoViewer', 'gl_Stripe_Wipe',
        'gl_swap', 'gl_Swirl', 'gl_WaterDrop', 'gl_windowblinds', 'gl_windowslice'
    ]

class VideoMerger:
    """视频合并器类，负责视频预处理和合并转场功能"""
    
    def __init__(self):
        """初始化视频合并器"""
        self.temp_files = []
    
    def cleanup(self):
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"清理临时文件 {temp_file} 失败: {e}")
        self.temp_files = []
    
    def preprocess_video(self, video_path, target_fps, target_width, target_height):
        """预处理单个视频文件
        
        Args:
            video_path: 输入视频路径
            target_fps: 目标帧率
            target_width: 目标宽度
            target_height: 目标高度
            
        Returns:
            str: 处理后的临时视频路径
        """
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        self.temp_files.append(temp_path)
        
        # 构建滤镜链
        filter_complex = []
        
        # 初始化滤镜前缀
        filters_prefix = ""
        
        # 1. 如果视频包含音频，先获取音频时长并对齐视频时长
        if has_audio(video_path):
            audio_duration = get_audio_duration(video_path)
            video_info = get_video_info(video_path)
            video_duration = video_info['duration'] if video_info else 0
            
            # 如果音频比视频长，使用tpad增加视频时长
            if audio_duration > video_duration:
                pad_duration = audio_duration - video_duration
                # 使用tpad在视频末尾添加填充，保持视频内容连续性
                filters_prefix = f"tpad=stop_mode=clone:stop_duration={pad_duration},"
        
        # 构建滤镜 - 应用音频对齐填充
        filter_complex.append(
            f'[0:v]settb=AVTB,fps=fps={target_fps},format=yuv420p,'
            f'{filters_prefix}'  # 音频对齐填充
            f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,'
            f'pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2[out]'
        )
        
        # 合并所有滤镜
        full_filter_complex = ";".join(filter_complex)
        
        # 构建FFmpeg命令
        command = ['ffmpeg', '-y', '-i', video_path]
        command.extend(['-filter_complex', full_filter_complex])
        command.extend(['-map', '[out]'])
        
        # 检查是否有音频
        if has_audio(video_path):
            command.extend(['-map', '0:a', '-c:a', 'aac', '-b:a', '192k'])
        else:
            command.extend(['-an'])
        
        # 添加视频编码器和其他参数 - 增强兼容性
        command.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
        command.extend(['-pix_fmt', 'yuv420p'])  # 确保使用广泛兼容的像素格式
        command.extend(['-movflags', '+faststart'])  # 允许视频在下载完成前开始播放
        command.extend(['-profile:v', 'high'])  # 使用更兼容的编码配置文件
        command.extend(['-color_range', 'tv', '-colorspace', 'bt709'])  # 标准色彩设置
        command.append(temp_path)
        
        # 执行命令
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        # 检查返回码
        if result.returncode != 0:
            print(f"视频预处理失败: {result.stderr}")
            raise ValueError(f"视频预处理失败: {result.stderr}")
        
        return temp_path
    
    def add_transition_padding(self, video_path, transition_duration):
        """为视频添加转场填充
        
        Args:
            video_path: 输入视频路径
            transition_duration: 转场时长
            
        Returns:
            str: 添加填充后的临时视频路径
        """
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        self.temp_files.append(temp_path)
        
        # 构建滤镜链 - 添加转场填充
        filter_complex = [
            f'[0:v]tpad=stop_mode=clone:stop_duration={transition_duration},format=yuv420p[out]'
        ]
        
        full_filter_complex = ";".join(filter_complex)
        
        # 构建FFmpeg命令
        command = ['ffmpeg', '-y', '-i', video_path]
        command.extend(['-filter_complex', full_filter_complex])
        command.extend(['-map', '[out]'])
        
        # 复制音频
        if has_audio(video_path):
            command.extend(['-map', '0:a', '-c:a', 'copy'])
        else:
            command.extend(['-an'])
        
        # 添加视频编码器和其他参数 - 增强兼容性
        command.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
        command.extend(['-pix_fmt', 'yuv420p'])  # 确保使用广泛兼容的像素格式
        command.extend(['-profile:v', 'high'])  # 使用更兼容的编码配置文件
        command.extend(['-color_range', 'tv', '-colorspace', 'bt709'])  # 标准色彩设置
        command.append(temp_path)
        
        # 执行命令
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"添加转场填充失败: {result.stderr}")
            raise ValueError(f"添加转场填充失败: {result.stderr}")
        
        return temp_path
    
    def merge_videos_with_transitions(self, video_paths, reference_video_index, target_fps, transition_type, transition_duration, output_path, device="cpu"):
        """合并视频并添加转场
        
        Args:
            video_paths: 视频路径列表
            reference_video_index: 参考视频索引（决定输出尺寸）
            target_fps: 目标帧率
            transition_type: 转场类型
            transition_duration: 转场时长（秒）
            output_path: 输出路径（相对于ComfyUI的output目录）
            device: 设备类型（cpu或cuda）
            
        Returns:
            str: 输出视频的完整路径
        """
        try:
            # 验证视频路径数量
            if len(video_paths) < 2:
                raise ValueError("至少需要2个视频文件进行合并")
            
            # 验证参考视频索引
            if reference_video_index < 0 or reference_video_index >= len(video_paths):
                raise ValueError(f"参考视频索引超出范围，有效范围：0-{len(video_paths)-1}")
            
            # 验证视频文件格式和存在性
            abs_paths = []
            for path in video_paths:
                abs_path = os.path.abspath(path.strip())
                
                if not abs_path.lower().endswith(tuple(get_supported_video_extensions())):
                    raise ValueError(f"文件 {abs_path} 不是支持的视频格式")
                
                if not os.path.isfile(abs_path):
                    raise ValueError(f"视频文件 {abs_path} 不存在")
                
                abs_paths.append(abs_path)
            
            # 获取参考视频的参数
            reference_info = get_video_info(abs_paths[reference_video_index])
            if reference_info is None:
                raise ValueError("无法获取参考视频信息")
            
            target_width = reference_info['width']
            target_height = reference_info['height']
            
            # 第一步：预处理所有视频
            print("开始预处理视频...")
            processed_videos = []
            processed_durations = []
            has_audio_list = []
            
            for i, video_path in enumerate(abs_paths):
                print(f"预处理视频 {i+1}/{len(abs_paths)}: {video_path}")
                
                # 预处理视频（统一帧率、尺寸和对齐音频视频时长）
                processed_video = self.preprocess_video(video_path, target_fps, target_width, target_height)
                processed_videos.append(processed_video)
                
                # 获取处理后的视频信息
                processed_info = get_video_info(processed_video)
                processed_durations.append(processed_info['duration'] if processed_info else 0)
                
                # 检查是否有音频
                has_audio_list.append(has_audio(video_path))
            
            # 第二步：为需要转场的视频添加转场填充（保留这一步确保转场动效可以正常加载）
            print("添加转场填充...")
            videos_with_padding = []
            
            for i, video_path in enumerate(processed_videos):
                # 对除了最后一个视频外的所有视频添加转场填充
                if i < len(processed_videos) - 1:
                    padded_video = self.add_transition_padding(video_path, transition_duration)
                    videos_with_padding.append(padded_video)
                else:
                    videos_with_padding.append(video_path)
            
            # 第三步：构建合并命令并执行
            print("开始合并视频并添加转场...")
            
            # 构建滤镜链
            filter_complex = []
            
            # 为每个视频创建输入标记
            for i in range(len(videos_with_padding)):
                filter_complex.append(f'[{i}:v]settb=AVTB,fps=fps={target_fps},format=yuv420p[v{i}]')
            
            # 构建视频转场链
            current_output = "v0"
            cumulative_duration = processed_durations[0]  # 从第一个视频的时长开始累计
            
            for i in range(1, len(videos_with_padding)):
                # 计算转场开始时间：设置为前一个视频的完整时长
                # 这样确保转场只会在第一个视频播放完毕后才开始
                transition_offset = cumulative_duration - transition_duration
                next_output = f"vout{i}"
                
                # 构建xfade滤镜参数，确保转场正确应用在视频之间
                filter_complex.append(
                    f'[{current_output}][v{i}]xfade=transition={transition_type}:duration={transition_duration}:offset={transition_offset}[{next_output}]'
                )
                
                current_output = next_output
                # 累加当前视频的实际时长，为下一个转场计算做准备
                if i < len(processed_durations):
                    cumulative_duration += processed_durations[i]
            
            # 处理音频：使用concat滤镜按顺序拼接音频流
            has_audio_any = any(has_audio_list)
            if has_audio_any:
                # 收集所有有音频的视频索引
                audio_indices = [i for i, has_audio_flag in enumerate(has_audio_list) if has_audio_flag]
                
                # 创建音频输入标记
                for i in audio_indices:
                    filter_complex.append(f'[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}]')
                
                # 构建音频concat滤镜
                if len(audio_indices) > 1:
                    # 构建音频输入列表
                    audio_inputs = ''.join([f'[a{i}]' for i in audio_indices])
                    filter_complex.append(f'{audio_inputs}concat=n={len(audio_indices)}:v=0:a=1[outa]')
                else:
                    # 只有一个音频流，直接重命名
                    filter_complex.append(f'[a{audio_indices[0]}]asplit[outa]')
            
            # 合并所有滤镜
            full_filter_complex = ";".join(filter_complex)
            
            # 构建FFmpeg命令
            command = ['ffmpeg', '-y']  # 添加-y参数自动覆盖输出文件
            
            # 设置硬件加速
            if device == "cuda":
                command.extend(['-hwaccel', 'cuda'])
            
            # 添加所有输入文件
            for path in videos_with_padding:
                command.extend(['-i', path])
            
            # 添加滤镜和视频映射
            command.extend(['-filter_complex', full_filter_complex])
            command.extend(['-map', f'[{current_output}]'])
            
            # 映射处理后的音频流
            if has_audio_any:
                command.extend(['-map', '[outa]'])
                # 设置音频编码器参数
                command.extend(['-c:a', 'aac', '-b:a', '192k'])
            else:
                # 如果没有音频，禁用音频轨道
                command.extend(['-an'])
            
            # 添加视频编码器参数 - 增强跨平台兼容性
            if device == "cuda":
                command.extend(['-c:v', 'h264_nvenc'])
                # NVENC特定优化参数
                command.extend(['-profile:v', 'high'])
            else:
                command.extend(['-c:v', 'libx264'])
                # 使用更兼容的编码配置文件
                command.extend(['-profile:v', 'high'])
            
            # 添加输出质量、性能参数和增强兼容性的设置
            command.extend(['-preset', 'medium', '-crf', '23'])
            command.extend(['-pix_fmt', 'yuv420p'])  # 确保使用广泛兼容的像素格式
            command.extend(['-movflags', '+faststart'])  # 允许视频在下载完成前开始播放
            command.extend(['-color_range', 'tv'])  # 设置标准色彩范围
            command.extend(['-colorspace', 'bt709'])  # 设置标准色彩空间
            
            # 检查output_path是否包含文件名
            if os.path.splitext(output_path)[1]:
                # 如果包含扩展名，则认为是完整文件路径
                output_file = output_path
                # 确保目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            else:
                # 如果不包含扩展名，则认为是目录路径
                os.makedirs(output_path, exist_ok=True)
                output_file = os.path.join(output_path, "merged_video.mp4")
            
            command.append(output_file)
            
            # 打印调试信息：显示完整的FFmpeg命令
            print("执行的FFmpeg命令:")
            print(' '.join(command))
            
            # 执行命令
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            print(f"视频合并成功完成！输出文件：{output_file}")
            return output_file
            
        except Exception as e:
            # 确保清理临时文件
            self.cleanup()
            raise ValueError(f"视频合并失败：{str(e)}")
        finally:
            # 清理临时文件
            self.cleanup()