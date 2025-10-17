# ComfyUI 多功能插件

这是一个自用的简易ComfyUI节点

## 功能特性

### 1. 图片转SVG节点 (ImageToSVG)

将输入的图片转换为SVG格式。该节点依赖于外部命令行工具 `potrace`，请确保已正确安装并配置其环境变量。

**输入参数：**
- `image`: 输入图片 (IMAGE类型)
- `output_filename_prefix`: 输出SVG文件的前缀名 (STRING类型，默认: `ComfyUI_WYQilin`)
- `output_path`: 输出SVG文件的目录 (STRING类型，如果为空则保存到系统临时目录)

**输出：**
- `SVG_DATA`: 生成的SVG数据 (STRING类型)
- `SVG_PATH`: 生成的SVG文件保存路径 (STRING类型)

**安装 `potrace` (重要!):
- **Windows:**
  1. 从 [Potrace官方网站](http://potrace.sourceforge.net/) 下载Windows版可执行文件 (例如 `potrace-X.X-win64.zip`)。
  2. 将下载的文件解压到一个方便的目录，例如 `C:\Program Files\Potrace`。
  3. 将 `potrace.exe` 所在的目录添加到系统的 `Path` 环境变量中。 (搜索"编辑系统环境变量" -> "环境变量" -> "系统变量"中的"Path" -> "编辑" -> "新建"并添加目录)
  4. **重启ComfyUI和您的终端/IDE** 以使环境变量生效。
- **Linux/macOS:**
  - 大多数Linux发行版可以通过包管理器安装：`sudo apt-get install potrace` (Debian/Ubuntu) 或 `sudo yum install potrace` (Fedora/CentOS)。
  - macOS可以使用Homebrew安装：`brew install potrace`。

### 2. 多图合并节点 (MultiImageMerger)

将任意大小的图片列表合并到一张图上。

**输入参数：**
- `images`: 图片列表 (支持图片批次和列表)
- `direction`: 拼接方向
  - `horizontal`: 横向拼接（保持高度相同）
  - `vertical`: 竖向拼接（保持宽度相同）
- `items_per_row`: 每行/列的图片数量 (1-10)

**输出：**
- `image`: 合并后的图片 (IMAGE类型)

**特性：**
- 自动调整图片尺寸以保持比例
- 支持网格布局
- 智能居中对齐
- 处理任意数量的输入图片

### 3. JSON提取器节点 (JSONExtractor)

从JSON文本中提取指定key的值，支持多层级访问。

**输入参数：**
- `json_text`: JSON格式的文本 (STRING类型)
- `extract_key`: 要提取的key路径 (STRING类型)

**输出：**
- 提取到的值 (STRING类型)

**Key路径语法：**
- 简单key: `"name"`
- 嵌套对象: `"user.profile.age"`
- 数组索引: `"users.0.name"`
- 复合路径: `"data.items.2.title"`

**示例：**
```json
{
  "users": [
    {
      "name": "张三",
      "profile": {
        "age": 25,
        "city": "北京"
      }
    }
  ]
}
```

- `"users.0.name"` → `"张三"`
- `"users.0.profile.city"` → `"北京"`
- `"users.0.profile"` → 完整的profile对象

### 4. 视频合并转场节点 (VideoMergerWithTransitions)

合并多个视频文件并在视频之间添加转场效果。支持多种转场类型，包括xfade和gl-transition两大类共111种转场效果。

**输入参数：**
- `video_paths`: 视频文件路径列表，每行一个路径 (STRING类型，多行文本)
- `reference_video_index`: 参考视频的索引（从0开始），决定输出视频的尺寸 (INT类型，默认: 0)
- `target_fps`: 目标帧率，所有视频将转换为该帧率 (FLOAT类型，默认: 30.0)
- `transition`: 转场效果类型 (选择框，默认: "fade")
- `transition_duration`: 转场持续时间，单位秒，最大值为3秒 (FLOAT类型，默认: 1.0)
- `output_path`: 相对于ComfyUI的output目录的输出路径 (STRING类型，默认: "merged_videos")
- `device`: 处理设备，可选cpu或cuda (STRING类型，默认: "cpu")

**输出：**
- `merged_video_path`: 合并后的视频文件路径 (STRING类型)

**主要特性：**
- 支持111种转场效果（47种xfade基本转场和64种gl-transition高级转场）
- 自动调整所有视频到统一的尺寸和帧率
- 精确的转场时间计算，确保转场在视频之间自然过渡
- 完整的音频处理，确保音视频同步
- 支持批量视频合并（至少需要2个视频文件）

**转场效果分类：**

1. **xfade基本转场效果**（共47种）
   - 淡入淡出：`fade`, `fadeblack`, `fadewhite`
   - 圆形效果：`circleclose`, `circleopen`, `rectclose`, `rectopen`
   - 滑动效果：`wipeleft`, `wiperight`, `wipeup`, `wipedown`
   - 覆盖效果：`coverleft`, `coverright`, `coverup`, `coverdown`
   - 揭示效果：`revealleft`, `revealright`, `revealup`, `revealdown`
   - 推入效果：`pushleft`, `pushright`, `pushup`, `pushdown`
   - 分割效果：`hblur`, `luma`, `pixelize`, `dodge`, `xor`, `glow`, `smoothleft`, `smoothright`, `smoothup`, `smoothdown`, `circlecrop`, `rectcrop`, `distance`, `blur`, `diagonalright`, `diagonalleft`, `hlslice`, `hrslice`, `vuslice`, `vdslice`

2. **gl-transition高级转场效果**（共64种）
   - 基本效果：`gl_angular`, `gl_Bars`, `gl_Blink`, `gl_Blobby`, `gl_Blocks`, `gl_BlockSize`
   - 翻页效果：`gl_BookFlip`, `gl_BouncySlide`, `gl_Circular`, `gl_ClockwiseSquares`
   - 立方体效果：`gl_cube`, `gl_Curves`
   - 扭曲效果：`gl_CrossZoom`, `gl_Directional`, `gl_Directional_bak`, `gl_DirectionalWarp`
   - 动态效果：`gl_Dreamy`, `gl_DreamyZoom`, `gl_Drip`, `gl_Droplet`
   - 高级效果：`gl_expo`, `gl_Expo`, `gl_Fade`, `gl_FadeColor`, `gl_FilmBurn`, `gl_Floor`
   - 几何效果：`gl_Geometry`, `gl_Glitch`, `gl_Heart`, `gl_Hexagonalize`, `gl_HorizontalSplit`
   - 创意效果：`gl_InvertedPageCurl`, `gl_LinearBlur`, `gl_Melt`, `gl_Morph`, `gl_Mosaic`
   - 复杂效果：`gl_MoveLeft`, `gl_MoveRight`, `gl_Multiply`, `gl_PageCurl`, `gl_Perspective`
   - 特殊效果：`gl_Pinch`, `gl_Pixelize`, `gl_PolarFunction`, `gl_PolkaDotsCurtain`, `gl_Radial`, `gl_RainbowCurtain`
   - 高级动态：`gl_Slice`, `gl_Slide`, `gl_Spiral`, `gl_Squareness`, `gl_Squares`
   - 变形效果：`gl_Squeeze`, `gl_StarWipe`, `gl_StereoViewer`, `gl_Teleport`, `gl_Triangles`
   - 高级翻转：`gl_Twirl`, `gl_VerticalSplit`, `gl_Wave`

## 安装方法

1. 确保已安装Python依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 确保已安装必要的外部工具：
   - `potrace` 命令行工具（详见 `图片转SVG节点` 部分的安装说明）
   - `ffmpeg` 命令行工具（用于视频合并转场节点，需要确保已安装并配置环境变量）
3. 将插件文件夹复制到ComfyUI的 `custom_nodes` 目录下
4. 重启ComfyUI
5. 在节点菜单中找到：
   - `WYQilin/image` → `图片转SVG`
   - `WYQilin/image` → `多图合并`
   - `WYQilin/json` → `JSON提取器`
   - `WYQilin/video` → `视频合并转场`

## 使用示例

### 图片转SVG
1. 将图片连接到"图片转SVG"节点的 `image` 输入
2. (可选) 设置 `output_filename_prefix` 和 `output_path`
3. 获得SVG数据和保存路径

### 多图合并
1. 连接多个图片输出到"多图合并"节点的 `images` 输入
2. 选择拼接方向（横向/竖向）
3. 设置每行/列的图片数量
4. 获得合并后的图片

### JSON提取器
1. 将JSON文本连接到 `json_text` 输入
2. 在 `extract_key` 中输入要提取的路径（如：`data.items.0.title`）
3. 获得提取的值

### 视频合并转场
1. 在 `video_paths` 输入中按行输入视频文件路径（每行一个路径）
2. 选择参考视频索引（决定输出视频的尺寸）
3. 设置目标帧率
4. 选择转场效果类型（如fade、slideleft等）
5. 设置转场持续时间（0.1-3.0秒）
6. （可选）设置输出路径
7. 选择处理设备（CPU或CUDA）
8. 获得合并后的视频文件路径

## 错误处理

插件包含完善的错误处理机制：
- `potrace` 可执行文件未找到提示
- `ffmpeg` 可执行文件未找到提示
- JSON格式验证
- Key路径有效性检查
- 图片尺寸兼容性处理
- 视频格式和数量验证
- 如果JSON提取器中指定字段不存在，将返回"None"
- 详细的错误信息提示

## 技术细节

- 支持任意尺寸的输入图片
- 自动处理图片比例缩放
- 内存优化的图片处理
- 兼容ComfyUI的张量格式
- 支持中文和Unicode字符
- 视频尺寸和帧率自动统一
- 支持111种转场效果
- 音频和视频同步处理

## 测试

运行测试脚本验证功能：
```bash
python test_video_merger.py
```

## 版本信息

- 版本: 1.1.0
- 兼容: ComfyUI
- 依赖: torch, PIL, numpy, svgpathtools, ffmpeg (命令行工具)
