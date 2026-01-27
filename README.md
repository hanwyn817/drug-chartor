# Drug-Chartor

基于LLM的药品稳定性考察趋势图生成工具

## 功能简介

Drug-Chartor是一个自动化的药品稳定性数据处理和趋势图生成工具。它可以从原始的Word/Excel文档中提取稳定性数据，并生成交互式的趋势折线图。

### 主要功能

1. **文档转换**：自动将Excel和Word文档转换为CSV/HTML格式
2. **智能分析**：使用大语言模型（LLM）识别和提取稳定性趋势数据
3. **数据验证**：自动验证数据格式和完整性
4. **图表生成**：生成交互式HTML趋势图，支持多批次对比

## 安装

### 环境要求

- Python 3.8+
- Windows系统（文档转换依赖pywin32处理Office文档）
- macOS/Linux 可运行除文档转换外的流程（需先准备转换后的 CSV/HTML）
- OpenAI兼容的API（如OpenAI、Azure OpenAI、或本地部署的模型）
- uv（推荐，用于虚拟环境与依赖管理）

### 使用 uv 安装（推荐）

**uv 是一个快速的 Python 包管理器，比 pip 快 10-100 倍。**

#### 1. 安装 uv

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. 创建虚拟环境并安装依赖

```bash
uv venv
uv pip install -r requirements.txt
```

如需安装命令行脚本（`drug-chartor`）：

```bash
uv pip install -e .
```

#### 3. 运行应用

```bash
uv run python main.py
```

### 使用 pip 安装（传统方式）

如果不使用 uv，可以使用传统方式：

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# 或 source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 配置

### 设置API密钥

在项目根目录创建`.env`文件，并设置你的LLM API配置：

```env
# OpenAI兼容API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

#### 不同提供商的配置示例

**OpenAI官方：**
```env
OPENAI_API_KEY=sk-proj-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

**Azure OpenAI：**
```env
OPENAI_API_KEY=your_azure_api_key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_MODEL=gpt-4
```

**本地模型（如Ollama/LM Studio）：**
```env
OPENAI_API_KEY=not-needed
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama2
```

## 使用方法

### 基本用法

```bash
# 使用默认配置（input目录作为输入）
uv run python main.py
```

> macOS/Linux 仅支持跳过文档转换模式。请先在 Windows 上完成转换，
> 或自行准备 `output/` 中的 CSV/HTML，再使用 `--skip-conversion`。

也可使用模块或安装后的命令：

```bash
python -m drug_chartor
# 或
drug-chartor
```

### 指定输入目录

```bash
uv run python main.py --input ./my_data
```

### 跳过文档转换（如果已经转换过）

```bash
uv run python main.py --skip-conversion
```

### 完整参数示例

```bash
uv run python main.py \
  --input ./input \
  --output ./output \
  --extracted ./extracted \
  --charts ./charts \
  --api-url https://api.openai.com/v1 \
  --model gpt-4o
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 原始文档目录 | `./input` |
| `--output` | 转换后的HTML/CSV输出目录 | `./output` |
| `--extracted` | 提取的JSON数据目录 | `./extracted` |
| `--charts` | 生成的图表目录 | `./charts` |
| `--api-key` | LLM API密钥 | 从环境变量读取 |
| `--api-url` | LLM API基础URL | `https://api.openai.com/v1` |
| `--model` | 使用的模型名称 | `gpt-4o` |
| `--skip-conversion` | 跳过文档转换 | False |

## 数据格式要求

### 输入文件

- 支持的格式：`.xls`, `.xlsx`, `.xlsm`, `.xlsb` (Excel)，`.doc`, `.docx` (Word)
- 文件应包含稳定性考察数据，如不同时间点的检测结果

### 提取的数据结构

工具会提取以下信息：

1. **产品名称**：如"达格列净"
2. **批号**：格式为 `D5XXX-YY-ZZZ`，后面可追加 1–2 位字母/数字，如 `D5290-24-004`、`D5290-24-004M1`
3. **市场/标准**：如 CEP、EDMF、USDMF、国内标准等
4. **考察条件**：如 长期Ⅱ、长期ⅣB、加速等
5. **考察周期**：如 0月、3月、6月、9月、12月等
6. **检测项目和结果**：
   - 干燥失重
   - 水分
   - 有关物质
   - 杂质
   - 含量/含量测定

### 数据验证规则

- **批号格式**：必须符合 `D5XXX-YY-ZZZ`，后面可追加 1–2 位字母/数字
- **检测项目**：只保留有数值的项目，排除纯文本项目（如外观、红外等）
- **数值类型**：所有检测结果必须是数值型数据

## 输出说明

### 目录结构

处理完成后会生成以下目录：

```
drug-chartor/
├── input/              # 原始文档（用户输入）
├── output/             # 转换后的CSV/HTML文件
├── extracted/          # 提取的JSON数据
│   └── 产品名_批号.json
└── charts/             # 生成的交互式图表
    ├── 产品名_批号.html
    ├── 产品名_市场_条件_Nbatches.html
    └── chart_summary.txt
```

### 图表特性

- **交互式**：使用Plotly生成，支持缩放、平移、悬停查看数值
- **多批次对比**：同一产品、市场、条件的多个批次会自动合并到一张图
- **专业样式**：统一字体、网格、图例与悬浮提示，适合报告展示
- **数据可视化**：清晰展示各检测项目随时间的变化趋势

## 工作流程

1. **文档转换**：使用`office_document_processor.py`将Office文档转换为机器可读格式
2. **LLM分析**：上传转换后的文件到LLM，判断是否包含稳定性数据
3. **数据提取**：对包含稳定性数据的文件，提取结构化信息
4. **数据验证**：验证数据格式和完整性
5. **图表生成**：根据提取的数据生成交互式趋势图

> 说明：若已存在 `output/` 的 CSV/HTML，可使用 `--skip-conversion` 跳过转换步骤，直接分析与出图。

## 常见问题

### Q: 如何处理不包含稳定性数据的文件？

A: 工具会自动跳过这些文件，不会生成警告或错误。

### Q: 批号格式不正确怎么办？

A: 批号必须符合 `D5XXX-YY-ZZZ`，后面可追加 1–2 位字母/数字。如果批号格式不正确，该文件会被标记为无效。

### Q: 可以自定义检测项目吗？

A: 可以。在`config.py`中修改`ACCEPTED_TEST_ITEMS`和`REJECTED_TEST_ITEMS`列表。

### Q: 图表可以导出为图片吗？

A: 可以。在交互式图表右上角有下载按钮，可选择导出为PNG格式。

### Q: 处理大量文件需要多长时间？

A: 处理时间取决于文件数量和LLM API的响应速度。建议分批处理大量文件。

## 项目结构

```
drug-chartor/
├── office_document_processor.py  # Office文档处理模块
├── config.py                     # 配置管理
├── llm_analyzer.py               # LLM分析模块
├── data_extractor.py             # 数据提取和验证
├── chart_generator.py            # 图表生成
├── workflow.py                   # 工作流编排
├── main.py                       # CLI入口
├── __main__.py                   # python -m 入口
├── drug_chartor/                 # 包入口
├── requirements.txt              # 依赖列表
├── input/                        # 原始数据目录
├── output/                       # 转换文件输出
├── extracted/                    # 提取数据输出
└── charts/                       # 图表输出
```

## 技术栈

- **文档处理**：pywin32 (Windows COM)
- **LLM集成**：OpenAI API
- **数据处理**：pandas, numpy
- **图表生成**：plotly
- **配置管理**：python-dotenv
- **包管理**：uv (推荐) 或 pip

## 使用 uv 的优势

**uv 是一个快速的 Python 包管理器，由 Astral 开发。**

主要优势：
- **速度快 10-100 倍**：比 pip 快得多的依赖安装速度
- **自动虚拟环境管理**：无需手动创建和激活虚拟环境
- **更好的依赖解析**：更智能的依赖冲突解决
- **Rust 实现**：更低的资源占用
- **Python 版本管理**：内置支持多个 Python 版本

更多信息：参考 uv 官方文档

## 许可证

本项目仅供内部使用。

## 更新日志

### v1.0.0 (2025-01-27)
- 初始版本发布
- 支持Excel和Word文档转换
- 集成OpenAI兼容API
- 自动提取和验证稳定性数据
- 生成交互式HTML趋势图
- 支持多批次对比
