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
\`\`\`json
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
\`\`\`

- `"users.0.name"` → `"张三"`
- `"users.0.profile.city"` → `"北京"`
- `"users.0.profile"` → 完整的profile对象

## 安装方法

1. 确保已安装Python依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 确保已安装 `potrace` 命令行工具（详见 `图片转SVG节点` 部分的安装说明）。
3. 将插件文件夹复制到ComfyUI的 `custom_nodes` 目录下
4. 重启ComfyUI
5. 在节点菜单中找到：
   - `WYQilin/image` → `图片转SVG`
   - `WYQilin/image` → `多图合并`
   - `WYQilin/json` → `JSON提取器`

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

## 错误处理

插件包含完善的错误处理机制：
- `potrace` 可执行文件未找到提示
- JSON格式验证
- Key路径有效性检查
- 图片尺寸兼容性处理
- 如果JSON提取器中指定字段不存在，将返回“None”
- 详细的错误信息提示

## 技术细节

- 支持任意尺寸的输入图片
- 自动处理图片比例缩放
- 内存优化的图片处理
- 兼容ComfyUI的张量格式
- 支持中文和Unicode字符

## 测试

运行测试脚本验证功能：
\`\`\`bash
python scripts/test_plugin.py
\`\`\`

## 版本信息

- 版本: 1.0.0
- 兼容: ComfyUI
- 依赖: torch, PIL, numpy, svgpathtools
