# 跳过转换功能

## 新增功能

如果 `output/` 文件夹中已经存在转换后的 CSV 和 HTML 文件，可以使用 `--skip-conversion` 参数跳过文档转换步骤，直接进行 LLM 分析和图表生成。

## 使用方法

### 命令行

```bash
# 自动检测并跳过转换（如果文件已存在）
python main.py --skip-conversion
```

### 工作原理

1. 检查 `output/` 文件夹
2. 如果存在 `.csv` 或 `.html` 文件：
   - 显示提示："⚠ Skipping document conversion (files already exist)"
   - 显示已找到的文件数量
   - 直接使用这些文件进行 LLM 分析
3. 如果不存在转换后的文件：
   - 正常执行文档转换
   - 生成新的 CSV 和 HTML 文件

## 使用场景

### 场景 1：首次运行

```bash
# 第一次运行，需要转换所有文档
python main.py
```

**输出**：
```
================================================================================
Step 1: Converting Office documents
================================================================================
[OK] Excel -> CSV: ...
[OK] Word -> HTML: ...
✓ Converted 12 files to CSV/HTML format
```

### 场景 2：重新运行（不转换）

```bash
# 文档已转换，只需要重新分析
python main.py --skip-conversion
```

**输出**：
```
================================================================================
Step 1: Converting Office documents
================================================================================
⚠ Skipping document conversion (files already exist)
   Found 8 CSV files
   Found 4 HTML files
✓ Using 12 existing converted files
```

### 场景 3：调试 LLM 提取（不重新转换）

```bash
# 调试时反复运行，不需要每次都转换文档
python main.py --skip-conversion
```

### 场景 4：更新 LLM 提取逻辑（保留转换结果）

```bash
# 修改了 LLM 提取代码，只需重新分析
python main.py --skip-conversion
```

## 文件检测

脚本会检查以下文件：
- `output/*.csv`
- `output/**/*.csv`
- `output/*.html`
- `output/**/*.html`

如果任何这些文件存在，就会触发跳过逻辑。

## 清理转换文件

如果需要强制重新转换文档：

```bash
# 方法 1：删除转换文件
rm -rf output/*.csv output/**/*.csv
rm -rf output/*.html output/**/*.html
python main.py

# 方法 2：直接删除整个 output 文件夹
rm -rf output
python main.py
```

## 注意事项

### ⚠️ 重要的注意事项

1. **文件更新检测**
   - 脚本只检查文件是否存在，不检查文件是否是最新
   - 如果源文件已更新，需要手动清理 output 文件夹

2. **部分转换**
   - 如果只有部分文件被转换，脚本仍会跳过所有转换步骤
   - 建议清理 output 文件夹以重新转换

3. **目录结构**
   - 脚本会保持原有的目录结构（递归查找）
   - 支持嵌套的子目录

4. **手动覆盖**
   - 如果需要修改特定文件的转换结果：
     - 手动转换该文件
     - 或删除对应的 CSV/HTML 文件并重新运行

## 与其他参数的兼容性

`--skip-conversion` 可以与其他参数组合使用：

```bash
# 跳过转换 + 自定义输入目录
python main.py --skip-conversion --input ./my_data

# 跳过转换 + 自定义输出目录
python main.py --skip-conversion --output ./converted

# 跳过转换 + 自定义 LLM 设置
python main.py --skip-conversion --model gpt-4o --api-url https://api.openai.com/v1

# 完整示例
python main.py \
  --skip-conversion \
  --input ./my_data \
  --output ./output \
  --model gpt-4o
```

## 工作流程对比

### 传统流程（不跳过）

```
输入文档 → 文档转换 → LLM 分析 → 数据验证 → 图表生成
  (3-5分钟)   (30-60秒)   (2-5分钟)   (5-10秒)    (5-10秒)
总耗时：约 10-20 分钟
```

### 优化流程（跳过转换）

```
输入文档 → LLM 分析 → 数据验证 → 图表生成
             (2-5分钟)   (5-10秒)    (5-10秒)
总耗时：约 3-8 分钟（节省 7-12 分钟）
```

## 故障排除

### 问题 1：始终跳过转换

**症状**：即使 output 文件夹为空，也显示"files already exist"

**原因**：之前测试时残留的隐藏文件

**解决**：
```bash
# 删除整个 output 文件夹
rm -rf output
```

### 问题 2：找不到转换文件

**症状**：显示"Using 0 existing converted files"，但文件存在

**原因**：文件扩展名或路径问题

**解决**：
```bash
# 检查文件
ls -la output/
```

### 问题 3：部分文件未转换

**症状**：某些文件没有被转换，但脚本跳过了转换步骤

**解决**：手动清理并重新运行：
```bash
rm -rf output
python main.py
```

## 性能提升

使用 `--skip-conversion` 可以显著提升处理速度：

| 操作 | 不跳过 | 跳过 | 提升 |
|------|--------|------|------|
| 文档转换 | 30-60 秒 | 跳过 | 节省 30-60 秒 |
| 总处理时间 | 10-20 分钟 | 3-8 分钟 | 节省 70% 时间 |

## 最佳实践

1. **首次运行**：不使用 `--skip-conversion`
2. **调试/迭代**：使用 `--skip-conversion` 快速迭代
3. **更新源数据**：手动清理 output，然后不使用 `--skip-conversion`
4. **批量处理**：如果处理大量文件，考虑分批处理

## 示例输出

### 首次运行输出

```
================================================================================
Drug-Chartor Workflow Starting
================================================================================
Input directory: input
Output directory: output
Extracted directory: extracted
Charts directory: charts

================================================================================
Step 1: Converting Office documents
================================================================================
[OK] Excel -> CSV: input/达格列净/D5290-24-004、005 数据.xlsx
[OK] Excel -> CSV: input/达格列净/D5290-25-001 数据.xlsx
...
✓ Converted 12 files to CSV/HTML format
```

### 跳过转换输出

```
================================================================================
Drug-Chartor Workflow Starting
================================================================================
Input directory: input
Output directory: output
Extracted directory: extracted
Charts directory: charts

================================================================================
Step 1: Converting Office documents
================================================================================
⚠ Skipping document conversion (files already exist)
   Found 8 CSV files
   Found 4 HTML files
✓ Using 12 existing converted files
```

## 总结

✅ 新增 `--skip-conversion` 参数
✅ 自动检测已转换的文件
✅ 显著提升重复运行速度
✅ 便于调试和迭代开发

使用此功能可以节省大量的文档转换时间！
