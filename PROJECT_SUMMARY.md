# Drug-Chartor 项目实现总结

## 项目概述

Drug-Chartor 是一个基于LLM的AI智能体工作流项目，用于从药品稳定性考察的原始数据自动生成稳定性考察趋势图。

## 已完成的功能模块

### 1. 核心模块

#### config.py - 配置管理
- ✓ LLM API 配置（API密钥、基础URL、模型名称）
- ✓ 文件路径配置
- ✓ 数据验证规则（批号格式、检测项目）
- ✓ 图表样式配置

#### llm_analyzer.py - LLM分析模块
- ✓ 文件稳定性数据筛选
- ✓ 结构化数据提取
- ✓ 批量文件处理
- ✓ 重试机制和错误处理

#### data_extractor.py - 数据提取和验证
- ✓ 批号格式验证（D5XXX-YY-ZZZ）
- ✓ 检测项目过滤（数值型项目）
- ✓ 数据类型验证
- ✓ 数据分组和聚合

#### chart_generator.py - 图表生成
- ✓ 单批次图表生成
- ✓ 多批次对比图表生成
- ✓ 交互式HTML图表
- ✓ 自定义样式配置

#### workflow.py - 工作流编排
- ✓ 端到端工作流管理
- ✓ 文档转换流程
- ✓ LLM分析流程
- ✓ 数据验证流程
- ✓ 图表生成流程

#### main.py - CLI入口
- ✓ 命令行参数解析
- ✓ 用户友好的错误提示
- ✓ 使用说明和示例

### 2. 辅助模块

#### office_document_processor.py（已存在）
- ✓ Excel → CSV 转换
- ✓ Word → HTML 转换
- ✓ 批量文档处理

### 3. 文档和配置

- ✓ README.md - 完整项目文档
- ✓ QUICKSTART.md - 快速开始指南
- ✓ requirements.txt - 依赖列表
- ✓ .env.example - 配置示例
- ✓ .gitignore - Git忽略配置
- ✓ setup.py - 自动安装脚本
- ✓ LICENSE - MIT许可证

## 项目结构

```
drug-chartor/
├── config.py                      # 配置管理模块
├── llm_analyzer.py               # LLM分析模块
├── data_extractor.py             # 数据提取和验证模块
├── chart_generator.py            # 图表生成模块
├── workflow.py                   # 工作流编排模块
├── main.py                       # CLI入口点
├── office_document_processor.py   # Office文档处理（已存在）
├── setup.py                      # 自动安装脚本
├── requirements.txt              # 依赖列表
├── .env.example                  # 配置示例
├── .gitignore                    # Git忽略配置
├── README.md                     # 完整文档
├── QUICKSTART.md                 # 快速开始指南
├── LICENSE                       # MIT许可证
├── input/                        # 原始数据目录
├── output/                       # 转换文件输出目录
├── extracted/                    # 提取数据输出目录
└── charts/                       # 图表输出目录
```

## 工作流程

```
1. 用户输入原始数据
   ↓
2. 文档转换（Excel → CSV, Word → HTML）
   ↓
3. LLM分析
   ├─ 筛选：判断文件是否包含稳定性数据
   └─ 提取：提取结构化数据
   ↓
4. 数据验证
   ├─ 批号格式验证
   ├─ 检测项目过滤
   └─ 数据类型验证
   ↓
5. 图表生成
   ├─ 单批次图表
   └─ 多批次对比图表
   ↓
6. 输出结果
   ├─ 提取的JSON数据
   ├─ 交互式HTML图表
   └─ 处理摘要
```

## 技术栈

- **语言**：Python 3.8+
- **文档处理**：pywin32 (Windows COM)
- **LLM集成**：OpenAI API（兼容接口）
- **数据处理**：pandas, numpy
- **图表生成**：plotly
- **配置管理**：python-dotenv
- **CLI**：argparse

## 支持的功能特性

### 数据处理
- ✓ 自动识别稳定性数据文件
- ✓ 批号格式验证（D5XXX-YY-ZZZ）
- ✓ 自动过滤纯文本检测项目
- ✓ 支持多种检测项目（干燥失重、水分、有关物质等）

### 图表生成
- ✓ 交互式HTML图表
- ✓ 支持多批次对比
- ✓ 自动按市场/标准分组
- ✓ 响应式设计
- ✓ 可导出为PNG

### 用户友好
- ✓ 简单的CLI接口
- ✓ 详细的错误提示
- ✓ 完整的文档
- ✓ 自动安装脚本

## 使用示例

### 基本使用
```bash
python main.py
```

### 自定义输入
```bash
python main.py --input ./my_data
```

### 跳过转换
```bash
python main.py --skip-conversion
```

### 完整参数
```bash
python main.py \
  --input ./input \
  --output ./output \
  --charts ./charts \
  --model gpt-4o
```

## 数据格式

### 输入
- Excel: .xls, .xlsx, .xlsm, .xlsb
- Word: .doc, .docx

### 输出
- 转换文件: CSV, HTML
- 提取数据: JSON
- 图表: 交互式HTML

## 配置说明

### 环境变量
```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

### 数据验证规则
- 批号格式：^D5\d{3}-\d{2}-\d{3}$
- 接受的检测项目：干燥失重、水分、有关物质、杂质、含量
- 拒绝的检测项目：外观、红外、鉴别

## 已解决的问题

1. ✓ 类型注解兼容性问题（Union[str, Path]）
2. ✓ LLM API调用重试机制
3. ✓ 数据验证和错误处理
4. ✓ 多批次数据分组
5. ✓ 图表样式自定义

## 后续优化建议

1. **性能优化**
   - 并发处理多个文件
   - 缓存已分析的结果
   - 增量更新模式

2. **功能增强**
   - 支持更多图表类型（柱状图、热力图）
   - 添加数据导出功能（Excel、PDF）
   - 支持自定义检测项目映射

3. **用户体验**
   - Web UI界面
   - 实时进度显示
   - 数据预览功能

4. **质量保证**
   - 单元测试
   - 集成测试
   - CI/CD流程

## 测试建议

1. 使用少量测试数据验证基本功能
2. 检查提取数据的准确性
3. 验证图表的交互功能
4. 测试错误处理和边界情况

## 注意事项

1. 需要Windows系统运行（pywin32依赖）
2. 需要配置OpenAI兼容API
3. 建议先处理少量数据测试
4. 注意API调用成本

## 项目状态

✓ 所有核心功能已实现
✓ 文档完整
✓ 可以直接使用

## 许可证

MIT License
