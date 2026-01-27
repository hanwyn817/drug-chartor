"""
统一的Office文档处理器（完整修正版 v2）
- Excel：多Sheet，每Sheet一个CSV；自动有效范围(A1:last cell)
- Word：同时导出 TXT + HTML（HTML由Python生成，避免Word SaveAs2 导出被EDS加密）
- 批处理：单次启动Excel/Word，多文件读取

依赖：
  pip install pywin32
"""

from __future__ import annotations

import sys
import io

# 设置标准输出编码为 UTF-8，解决 Windows 中文乱码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Iterable, List, Tuple
import time
import csv
from datetime import datetime
import html as _html

try:
    import pythoncom
    import win32com.client as win32
except Exception:  # pragma: no cover - handled by runtime platform checks
    pythoncom = None
    win32 = None

_WIN32_AVAILABLE = (sys.platform == "win32") and (pythoncom is not None) and (win32 is not None)


def _require_windows(feature: str) -> None:
    if not _WIN32_AVAILABLE:
        raise RuntimeError(
            f"{feature} requires Windows with pywin32 installed. "
            f"Current platform: {sys.platform}. "
            "Use --skip-conversion or run on Windows."
        )

# -----------------------------
# Constants
# -----------------------------
# Excel constants
XL_BY_ROWS = 1
XL_BY_COLUMNS = 2
XL_PREVIOUS = 2
XL_FORMULAS = -4123
XL_VALUES = -4163
XL_CALC_MANUAL = -4135

# Word constants
WD_ALERTS_NONE = 0

# Sheet visibility (Excel XlSheetVisibility)
XL_SHEET_VISIBLE = -1


# -----------------------------
# Data model
# -----------------------------
@dataclass
class OfficeReadResult:
    ok: bool
    app: str
    path: str
    elapsed_ms: int = 0

    # content
    text: str = ""
    char_count: int = 0
    preview: str = ""

    # Excel meta
    excel_last_row: Optional[int] = None
    excel_last_col: Optional[int] = None
    excel_sheet: Optional[Union[int, str]] = None
    excel_sheet_name: Optional[str] = None
    excel_range_a1: Optional[str] = None

    # error
    error_type: str = ""
    error_message: str = ""


# -----------------------------
# Utils
# -----------------------------
def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return repr(x)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _a1_col(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA"""
    if n <= 0:
        return "A"
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _a1_addr(row: int, col: int) -> str:
    return f"{_a1_col(col)}{row}"


def _values_to_delimited_text(values: Any, sep: str = "\t", row_sep: str = "\n") -> str:
    if values is None:
        return ""
    if not isinstance(values, tuple):
        return _safe_str(values)

    lines: List[str] = []
    for row in values:
        if isinstance(row, tuple):
            lines.append(sep.join(_safe_str(c) for c in row))
        else:
            lines.append(_safe_str(row))
    return row_sep.join(lines)


def _values_to_rows(values: Any) -> List[List[str]]:
    if values is None:
        return []
    if not isinstance(values, tuple):
        return [[_safe_str(values)]]

    rows: List[List[str]] = []
    for row in values:
        if isinstance(row, tuple):
            rows.append([_safe_str(c) for c in row])
        else:
            rows.append([_safe_str(row)])
    return rows


def safe_filename(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def unique_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def export_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", errors="replace")


def export_html(path: Union[str, Path], html_text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # HTML 更推荐纯 utf-8（不带 BOM）
    p.write_text(html_text, encoding="utf-8", errors="replace")


def write_csv_utf8(path: Union[str, Path], rows: List[List[str]]) -> None:
    """
    CSV 用 utf-8-sig，确保 Excel 双击打开不乱码
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)


def iter_office_files(root: Path) -> Iterable[Path]:
    exts = {".xls", ".xlsx", ".xlsm", ".xlsb", ".doc", ".docx"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _html_escape(s: str) -> str:
    return _html.escape(s, quote=False)


def _clean_word_text_for_html(s: str) -> str:
    """
    Word Range.Text 常带 '\r' 结尾（段落标记），且表格单元格末尾带 '\x07'
    """
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = s.replace("\x07", "")  # end-of-cell
    s = s.replace("\x0b", "\n")  # line break
    return s.strip()


def _heading_level_from_style_name(style_name: str) -> Optional[int]:
    """
    从样式名推断标题级别。
    兼容：Heading 1 / 标题 1 / 标题1 / 标题 2 等。
    """
    if not style_name:
        return None

    s = style_name.strip().lower()
    # English: heading 1..9
    if "heading" in s:
        for n in range(1, 7):
            if f"heading {n}" in s or f"heading{n}" in s:
                return n

    # Chinese: 标题1 / 标题 1
    if "标题" in style_name:
        for n in range(1, 7):
            if f"标题{n}" in style_name or f"标题 {n}" in style_name:
                return n

    return None


def _table_to_html(table) -> str:
    """
    将 Word Table 转为 HTML table
    注意：合并单元格(rowspan/colspan) 处理较复杂，此处采用“按可见单元格文本输出”的简化策略。
    """
    # Word COM 的 Rows/Columns 是 1-based
    try:
        n_rows = int(table.Rows.Count)
        n_cols = int(table.Columns.Count)
    except Exception:
        # 兜底：无法计数则直接返回空
        return ""

    lines: List[str] = []
    lines.append('<table border="1" cellspacing="0" cellpadding="4">')
    for r in range(1, n_rows + 1):
        lines.append("<tr>")
        for c in range(1, n_cols + 1):
            try:
                cell_text = table.Cell(r, c).Range.Text
            except Exception:
                cell_text = ""
            cell_text = _clean_word_text_for_html(cell_text)
            lines.append(f"<td>{_html_escape(cell_text)}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _doc_to_html(doc, title: str = "") -> str:
    """
    从 Word Document 生成 HTML（不使用 SaveAs2，避免导出被EDS加密）
    - 段落按顺序输出
    - 表格按在文档中的位置插入，避免重复输出
    """
    # 收集表格范围，用于判断段落是否落在表格内
    tables_info: List[Tuple[int, int, Any]] = []
    try:
        for t in doc.Tables:
            try:
                start = int(t.Range.Start)
                end = int(t.Range.End)
                tables_info.append((start, end, t))
            except Exception:
                continue
        tables_info.sort(key=lambda x: x[0])
    except Exception:
        tables_info = []

    emitted_table_starts = set()

    body: List[str] = []
    body.append("<!doctype html>")
    body.append('<html lang="zh-CN">')
    body.append("<head>")
    body.append('<meta charset="utf-8">')
    body.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    body.append(f"<title>{_html_escape(title) if title else 'document'}</title>")
    # 轻量样式：AI 解析不依赖样式，但便于人看
    body.append(
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;line-height:1.55;margin:24px;}"
        "table{border-collapse:collapse;margin:12px 0;}"
        "td,th{border:1px solid #999;padding:4px 6px;vertical-align:top;}"
        "pre{white-space:pre-wrap;}"
        "</style>"
    )
    body.append("</head>")
    body.append("<body>")

    # 逐段落输出；遇到表格范围时插入表格并跳过表格内段落（避免重复）
    try:
        paras = doc.Paragraphs
        for i in range(1, int(paras.Count) + 1):
            p = paras(i)
            rng = p.Range
            try:
                p_start = int(rng.Start)
            except Exception:
                p_start = -1

            # 如果该段落起点落在某个表格范围内，输出表格（仅一次），然后 continue
            in_table = False
            for t_start, t_end, t in tables_info:
                if t_start <= p_start < t_end:
                    in_table = True
                    if t_start not in emitted_table_starts:
                        emitted_table_starts.add(t_start)
                        table_html = _table_to_html(t)
                        if table_html:
                            body.append(table_html)
                    break
            if in_table:
                continue

            # 正常段落
            try:
                raw = rng.Text
            except Exception:
                raw = ""
            text = _clean_word_text_for_html(raw)
            if not text:
                # 空行：保留一个换行（用 <br>）
                body.append("<br>")
                continue

            # 判断样式是否为标题
            try:
                style_name = rng.Style.NameLocal
            except Exception:
                try:
                    style_name = str(rng.Style)
                except Exception:
                    style_name = ""

            h_level = _heading_level_from_style_name(style_name)
            if h_level is not None:
                body.append(f"<h{h_level}>{_html_escape(text)}</h{h_level}>")
            else:
                # 保留换行：用 <pre> 适合包含多行的段落（例如地址块/签名块）
                if "\n" in text:
                    body.append(f"<pre>{_html_escape(text)}</pre>")
                else:
                    body.append(f"<p>{_html_escape(text)}</p>")

    except Exception as e:
        body.append(f"<pre>Failed to parse document paragraphs: {_html_escape(_safe_str(e))}</pre>")

    body.append("</body>")
    body.append("</html>")
    return "\n".join(body)


# -----------------------------
# One-off sessions (single file)
# -----------------------------
class _ExcelSession:
    def __init__(self, visible: bool = False):
        self.visible = visible
        self.excel = None

    def __enter__(self):
        pythoncom.CoInitialize()
        self.excel = win32.DispatchEx("Excel.Application")
        self.excel.Visible = bool(self.visible)
        self.excel.DisplayAlerts = False
        self.excel.AskToUpdateLinks = False
        self.excel.EnableEvents = False
        self.excel.ScreenUpdating = False
        try:
            self.excel.Calculation = XL_CALC_MANUAL
        except Exception:
            pass
        try:
            self.excel.AutomationSecurity = 3
        except Exception:
            pass
        return self.excel

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.excel is not None:
                self.excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


class _WordSession:
    def __init__(self, visible: bool = False):
        self.visible = visible
        self.word = None

    def __enter__(self):
        pythoncom.CoInitialize()
        self.word = win32.DispatchEx("Word.Application")
        self.word.Visible = bool(self.visible)
        self.word.DisplayAlerts = WD_ALERTS_NONE
        return self.word

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.word is not None:
                self.word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


def _find_last_cell(ws, *, look_in: int) -> Tuple[int, int]:
    last_row_cell = ws.Cells.Find(
        What="*",
        After=ws.Cells(1, 1),
        LookIn=look_in,
        LookAt=1,
        SearchOrder=XL_BY_ROWS,
        SearchDirection=XL_PREVIOUS,
        MatchCase=False,
    )
    last_col_cell = ws.Cells.Find(
        What="*",
        After=ws.Cells(1, 1),
        LookIn=look_in,
        LookAt=1,
        SearchOrder=XL_BY_COLUMNS,
        SearchDirection=XL_PREVIOUS,
        MatchCase=False,
    )
    if last_row_cell is None or last_col_cell is None:
        return 1, 1
    return int(last_row_cell.Row), int(last_col_cell.Column)


# -----------------------------
# Read (single file)
# -----------------------------
def read_excel_text(
    path: str,
    sheet: Union[int, str] = 1,
    range_a1: Optional[str] = None,
    *,
    auto_bounds: bool = True,
    look_in: str = "formulas",
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    sep: str = "\t",
    row_sep: str = "\n",
    preview_chars: int = 800,
    visible: bool = False,
    password: Optional[str] = None,
) -> OfficeReadResult:
    _require_windows("Excel document conversion")
    t0 = time.time()
    wb = None
    try:
        with _ExcelSession(visible=visible) as excel:
            open_kwargs = dict(
                Filename=path,
                ReadOnly=True,
                UpdateLinks=0,
                IgnoreReadOnlyRecommended=True,
                AddToMru=False,
            )
            if password:
                open_kwargs["Password"] = password

            wb = excel.Workbooks.Open(**open_kwargs)
            ws = wb.Worksheets(sheet)
            sheet_name = ws.Name

            last_row = last_col = None
            effective_a1 = None

            if range_a1:
                rng = ws.Range(range_a1)
                effective_a1 = range_a1
            else:
                if auto_bounds:
                    lookin_const = XL_FORMULAS if look_in.lower() == "formulas" else XL_VALUES
                    lr, lc = _find_last_cell(ws, look_in=lookin_const)
                    if max_rows is not None:
                        lr = min(lr, int(max_rows))
                    if max_cols is not None:
                        lc = min(lc, int(max_cols))
                    rng = ws.Range(ws.Cells(1, 1), ws.Cells(lr, lc))
                    last_row, last_col = lr, lc
                    effective_a1 = f"A1:{_a1_addr(lr, lc)}"
                else:
                    rng = ws.Range("A1")
                    last_row, last_col = 1, 1
                    effective_a1 = "A1"

            values = rng.Value
            text = _normalize_newlines(_values_to_delimited_text(values, sep=sep, row_sep=row_sep))

            elapsed_ms = int((time.time() - t0) * 1000)
            return OfficeReadResult(
                ok=True,
                app="excel",
                path=path,
                elapsed_ms=elapsed_ms,
                text=text,
                char_count=len(text),
                preview=text[:preview_chars],
                excel_last_row=last_row,
                excel_last_col=last_col,
                excel_sheet=sheet,
                excel_sheet_name=sheet_name,
                excel_range_a1=effective_a1,
            )
    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        return OfficeReadResult(
            ok=False,
            app="excel",
            path=path,
            elapsed_ms=elapsed_ms,
            error_type=type(e).__name__,
            error_message=_safe_str(e),
        )
    finally:
        try:
            if wb is not None:
                wb.Close(SaveChanges=False)
        except Exception:
            pass


def read_word_text(
    path: str,
    *,
    preview_chars: int = 800,
    visible: bool = False,
    password: Optional[str] = None,
) -> OfficeReadResult:
    _require_windows("Word document conversion")
    t0 = time.time()
    doc = None
    try:
        with _WordSession(visible=visible) as word:
            open_kwargs = dict(
                FileName=path,
                ReadOnly=True,
                AddToRecentFiles=False,
                ConfirmConversions=False,
            )
            if password:
                open_kwargs["PasswordDocument"] = password

            doc = word.Documents.Open(**open_kwargs)
            text = _normalize_newlines(doc.Content.Text or "")

            elapsed_ms = int((time.time() - t0) * 1000)
            return OfficeReadResult(
                ok=True,
                app="word",
                path=path,
                elapsed_ms=elapsed_ms,
                text=text,
                char_count=len(text),
                preview=text[:preview_chars],
            )
    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        return OfficeReadResult(
            ok=False,
            app="word",
            path=path,
            elapsed_ms=elapsed_ms,
            error_type=type(e).__name__,
            error_message=_safe_str(e),
        )
    finally:
        try:
            if doc is not None:
                doc.Close(SaveChanges=False)
        except Exception:
            pass


def read_office_text(path: str, **kwargs) -> OfficeReadResult:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
        return read_excel_text(path, **kwargs)
    if suffix in {".doc", ".docx"}:
        return read_word_text(path, **kwargs)
    return OfficeReadResult(
        ok=False,
        app="unknown",
        path=path,
        error_type="UnsupportedFileType",
        error_message=f"Unsupported extension: {suffix}",
    )


# -----------------------------
# Batch readers
# -----------------------------
class ExcelBatchReader:
    def __init__(self, *, visible: bool = False, disable_macros: bool = True):
        self.visible = visible
        self.disable_macros = disable_macros
        self.excel = None

    def open(self) -> "ExcelBatchReader":
        _require_windows("Excel batch conversion")
        pythoncom.CoInitialize()
        self.excel = win32.DispatchEx("Excel.Application")
        self.excel.Visible = bool(self.visible)
        self.excel.DisplayAlerts = False
        self.excel.AskToUpdateLinks = False
        self.excel.EnableEvents = False
        self.excel.ScreenUpdating = False
        try:
            self.excel.Calculation = XL_CALC_MANUAL
        except Exception:
            pass
        if self.disable_macros:
            try:
                self.excel.AutomationSecurity = 3
            except Exception:
                pass
        return self

    def close(self) -> None:
        try:
            if self.excel is not None:
                self.excel.Quit()
        except Exception:
            pass
        finally:
            self.excel = None
            pythoncom.CoUninitialize()

    def _require_open(self) -> None:
        if self.excel is None:
            raise RuntimeError("ExcelBatchReader is not opened. Call .open() first.")

    def list_sheets(self, path: str, *, visible_only: bool = True) -> Tuple[bool, List[str], str]:
        self._require_open()
        wb = None
        try:
            wb = self.excel.Workbooks.Open(
                Filename=path,
                ReadOnly=True,
                UpdateLinks=0,
                IgnoreReadOnlyRecommended=True,
                AddToMru=False,
            )
            names: List[str] = []
            for ws in wb.Worksheets:
                try:
                    is_visible = (ws.Visible == XL_SHEET_VISIBLE)
                except Exception:
                    is_visible = True
                if (not visible_only) or is_visible:
                    names.append(ws.Name)
            return True, names, ""
        except Exception as e:
            return False, [], _safe_str(e)
        finally:
            try:
                if wb is not None:
                    wb.Close(SaveChanges=False)
            except Exception:
                pass

    def read_cells_sheet(
        self,
        path: str,
        *,
        sheet: Union[int, str],
        auto_bounds: bool = True,
        look_in: str = "formulas",
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
    ) -> Tuple[bool, Any, str, Optional[str], Optional[int], Optional[int], Optional[str]]:
        self._require_open()
        wb = None
        try:
            wb = self.excel.Workbooks.Open(
                Filename=path,
                ReadOnly=True,
                UpdateLinks=0,
                IgnoreReadOnlyRecommended=True,
                AddToMru=False,
            )
            ws = wb.Worksheets(sheet)
            sheet_name = ws.Name

            if auto_bounds:
                lookin_const = XL_FORMULAS if look_in.lower() == "formulas" else XL_VALUES
                last_row, last_col = _find_last_cell(ws, look_in=lookin_const)
                if max_rows is not None:
                    last_row = min(last_row, int(max_rows))
                if max_cols is not None:
                    last_col = min(last_col, int(max_cols))
                rng = ws.Range(ws.Cells(1, 1), ws.Cells(last_row, last_col))
                effective_a1 = f"A1:{_a1_addr(last_row, last_col)}"
            else:
                rng = ws.Range("A1")
                effective_a1 = "A1"
                last_row, last_col = 1, 1

            values = rng.Value
            return True, values, "", effective_a1, last_row, last_col, sheet_name
        except Exception as e:
            return False, None, _safe_str(e), None, None, None, None
        finally:
            try:
                if wb is not None:
                    wb.Close(SaveChanges=False)
            except Exception:
                pass


class WordBatchReader:
    def __init__(self, *, visible: bool = False):
        self.visible = visible
        self.word = None

    def open(self) -> "WordBatchReader":
        _require_windows("Word batch conversion")
        pythoncom.CoInitialize()
        self.word = win32.DispatchEx("Word.Application")
        self.word.Visible = bool(self.visible)
        self.word.DisplayAlerts = WD_ALERTS_NONE
        return self

    def close(self) -> None:
        try:
            if self.word is not None:
                self.word.Quit()
        except Exception:
            pass
        finally:
            self.word = None
            pythoncom.CoUninitialize()

    def _require_open(self) -> None:
        if self.word is None:
            raise RuntimeError("WordBatchReader is not opened. Call .open() first.")

    def open_doc(self, path: str):
        self._require_open()
        return self.word.Documents.Open(
            FileName=path,
            ReadOnly=True,
            AddToRecentFiles=False,
            ConfirmConversions=False,
        )

    def read_text(self, path: str, *, preview_chars: int = 800) -> OfficeReadResult:
        self._require_open()
        t0 = time.time()
        doc = None
        try:
            doc = self.open_doc(path)
            text = _normalize_newlines(doc.Content.Text or "")
            elapsed_ms = int((time.time() - t0) * 1000)
            return OfficeReadResult(
                ok=True,
                app="word",
                path=path,
                elapsed_ms=elapsed_ms,
                text=text,
                char_count=len(text),
                preview=text[:preview_chars],
            )
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return OfficeReadResult(
                ok=False,
                app="word",
                path=path,
                elapsed_ms=elapsed_ms,
                error_type=type(e).__name__,
                error_message=_safe_str(e),
            )
        finally:
            try:
                if doc is not None:
                    doc.Close(SaveChanges=False)
            except Exception:
                pass

    def build_html(self, path: str) -> Tuple[bool, str, str]:
        """
        读取 Word 并生成 HTML（不落地保存；由调用方用 Python 写入）
        返回：(ok, html_text, error_message)
        """
        self._require_open()
        doc = None
        try:
            doc = self.open_doc(path)
            title = Path(path).name
            html_text = _doc_to_html(doc, title=title)
            return True, html_text, ""
        except Exception as e:
            return False, "", _safe_str(e)
        finally:
            try:
                if doc is not None:
                    doc.Close(SaveChanges=False)
            except Exception:
                pass


# -----------------------------
# High-level processing APIs
# -----------------------------
def process_single_document(
    file_path: str,
    *,
    export: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    excel_sheets: Union[str, int, List[Union[str, int]], None] = "all",
    preview_chars: int = 300,
    export_excel_csv: bool = True,
    export_word_txt: bool = True,
    export_word_html: bool = True,
    visible_only_sheets: bool = True,
) -> OfficeReadResult:
    _require_windows("Office document conversion")
    p = Path(file_path)
    suffix = p.suffix.lower()

    # ---------- Excel ----------
    if suffix in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
        res = read_excel_text(
            file_path,
            sheet=1,
            auto_bounds=True,
            look_in="formulas",
            preview_chars=preview_chars,
        )

        if export and res.ok and output_dir is not None and export_excel_csv:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            excel = ExcelBatchReader(visible=False).open()
            try:
                if excel_sheets is None:
                    sheets_to_process: List[Union[str, int]] = [1]
                elif isinstance(excel_sheets, str) and excel_sheets.lower() == "all":
                    ok_s, sheet_names, err_s = excel.list_sheets(file_path, visible_only=visible_only_sheets)
                    if not ok_s:
                        print(f"[FAIL] list sheets failed: {p.name} - {err_s}")
                        sheets_to_process = [1]
                    else:
                        sheets_to_process = sheet_names
                elif isinstance(excel_sheets, (int, str)):
                    sheets_to_process = [excel_sheets]
                else:
                    sheets_to_process = list(excel_sheets)

                for sh in sheets_to_process:
                    ok, values, err, effective_a1, lr, lc, sheet_name = excel.read_cells_sheet(
                        file_path,
                        sheet=sh,
                        auto_bounds=True,
                        look_in="formulas",
                    )
                    if ok and values is not None:
                        out_path = out_dir / safe_filename(
                            f"{p.stem}__{sheet_name}__{unique_stamp()}_excel.csv"
                        )
                        rows = _values_to_rows(values)
                        write_csv_utf8(out_path, rows)
                        print(f"[OK] Exported CSV: {out_path} (range={effective_a1}, last={lr},{lc})")
                    else:
                        print(f"[FAIL] Export CSV failed: {p.name} / {sh} - {err}")
            finally:
                excel.close()

        return res

    # ---------- Word ----------
    if suffix in {".doc", ".docx"}:
        res = read_word_text(file_path, preview_chars=preview_chars)

        if export and res.ok and output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if export_word_txt:
                out_txt = out_dir / safe_filename(f"{p.stem}__{unique_stamp()}_word.txt")
                export_text(out_txt, res.text)
                print(f"[OK] Exported TXT: {out_txt}")

            if export_word_html:
                word = WordBatchReader(visible=False).open()
                try:
                    ok_h, html_text, err_h = word.build_html(file_path)
                    out_html = out_dir / safe_filename(f"{p.stem}__{unique_stamp()}_word.html")
                    if ok_h:
                        export_html(out_html, html_text)
                        print(f"[OK] Exported HTML (python-generated): {out_html}")
                    else:
                        print(f"[FAIL] Build HTML failed: {p.name} - {err_h}")
                finally:
                    word.close()

        return res

    return OfficeReadResult(
        ok=False,
        app="unknown",
        path=file_path,
        error_type="UnsupportedFileType",
        error_message=f"Unsupported extension: {suffix}",
    )


def process_documents_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    export_format: str = "both",  # 'excel' | 'word' | 'both'
    export_excel_csv: bool = True,
    export_word_txt: bool = True,  # 现在默认为True，但我们将在函数中忽略这个选项
    export_word_html: bool = True,
    visible_only_sheets: bool = True,
) -> None:
    _require_windows("Office document conversion")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_reader = ExcelBatchReader(visible=False).open()
    word_reader = WordBatchReader(visible=False).open()

    try:
        for p in iter_office_files(in_dir):
            suffix = p.suffix.lower()

            # 计算相对于输入目录的路径，用于保持目录结构
            rel_path = p.relative_to(in_dir)

            # ---------- Excel ----------
            if suffix in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
                if export_format not in {"excel", "both"}:
                    continue

                ok_s, sheet_names, err_s = excel_reader.list_sheets(str(p), visible_only=visible_only_sheets)
                if not ok_s:
                    print(f"[FAIL] Excel list sheets failed: {p.name} - {err_s}")
                    continue

                for sh_name in sheet_names:
                    t0 = time.time()
                    ok, values, err, effective_a1, lr, lc, real_name = excel_reader.read_cells_sheet(
                        str(p),
                        sheet=sh_name,
                        auto_bounds=True,
                        look_in="formulas",
                    )
                    elapsed_ms = int((time.time() - t0) * 1000)

                    if not ok:
                        print(f"[FAIL] Excel sheet read failed: {p.name} / {sh_name} - {err}")
                        continue

                    if export_excel_csv:
                        out_name = safe_filename(f"{p.stem}__{real_name}__{unique_stamp()}_excel.csv")
                        # 创建对应的输出子目录结构
                        output_subdir = out_dir / rel_path.parent
                        output_subdir.mkdir(parents=True, exist_ok=True)
                        out_path = output_subdir / out_name
                        rows = _values_to_rows(values)
                        write_csv_utf8(out_path, rows)
                        print(
                            f"[OK] Excel -> CSV: {p.name} / {real_name} "
                            f"({elapsed_ms}ms, range={effective_a1}, last={lr},{lc}) -> {out_path}"
                        )

            # ---------- Word ----------
            elif suffix in {".doc", ".docx"}:
                if export_format not in {"word", "both"}:
                    continue

                # TXT：读一次，复用写出
                res = word_reader.read_text(str(p), preview_chars=200)
                if not res.ok:
                    print(f"[FAIL] Word read failed: {p.name} - {res.error_message}")
                    continue

                # 创建对应的输出子目录结构
                output_subdir = out_dir / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                # # 注释掉Word导出TXT的功能
                # if export_word_txt:
                #     out_txt = output_subdir / safe_filename(f"{p.stem}__{unique_stamp()}_word.txt")
                #     export_text(out_txt, res.text)
                #     print(f"[OK] Word -> TXT: {p.name} ({res.elapsed_ms}ms) -> {out_txt}")

                if export_word_html:
                    ok_h, html_text, err_h = word_reader.build_html(str(p))
                    out_html = output_subdir / safe_filename(f"{p.stem}__{unique_stamp()}_word.html")
                    if ok_h:
                        export_html(out_html, html_text)
                        print(f"[OK] Word -> HTML (python-generated): {p.name} -> {out_html}")
                    else:
                        print(f"[FAIL] Word build HTML failed: {p.name} - {err_h}")

    finally:
        try:
            word_reader.close()
        finally:
            excel_reader.close()


def print_result(res: OfficeReadResult, label: str) -> None:
    print("=" * 80)
    print(f"LABEL      : {label}")
    print(f"APP        : {res.app}")
    print(f"PATH       : {res.path}")
    print(f"OK         : {res.ok}")
    print(f"ELAPSED(ms): {res.elapsed_ms}")
    if res.ok:
        print(f"CHARS      : {res.char_count}")
        print("PREVIEW:")
        print(res.preview)
        if res.app == "excel":
            print(f"SHEET      : {res.excel_sheet} ({res.excel_sheet_name})")
            print(f"RANGE      : {res.excel_range_a1} (last={res.excel_last_row},{res.excel_last_col})")
    else:
        print(f"ERROR_TYPE : {res.error_type}")
        print(f"ERROR_MSG  : {res.error_message}")


if __name__ == "__main__":
    pass
