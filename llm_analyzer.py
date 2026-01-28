"""
LLM Analyzer module for extracting stability data (OpenAI-compatible)

Key improvements vs original:
- Robust YES/NO screening parsing (reduces false negatives)
- Extraction uses response_format={"type":"json_object"} to force strict JSON
- Stronger JSON cleaning / fallback extraction
- Encoding fallback for reading files (utf-8-sig/utf-8/gb18030)
- Avoid hard-truncation at 20k chars; uses chunked extraction with merge for large docs
- Safer ThreadPoolExecutor usage + cleaner timeout handling
- Extraction temperature forced to 0 for stability
- Stronger schema validation
"""

import json
import time
import threading
import concurrent.futures
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from openai import OpenAI
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SCREENING,
    LLM_TIMEOUT_EXTRACTION,
    MAX_RETRIES,
    RETRY_DELAY,
    EXTRACTED_DIR,
)


class LLMAnalyzer:
    """Analyzer for extracting stability data using LLM"""

    SCREENING_PROMPT = """你正在分析一份药学/制药文档。请判断该文档是否包含药物稳定性趋势数据。

稳定性趋势数据通常包含：
- 多个时间点的检测结果（如 0、3、6、9、12、18、24、36、48 月等）
- 多个检测项目的数值结果（如水分、有关物质、含量等）
- 批号类似 D5XXX-YY-ZZZ（可有 1-3 位字母或数字后缀，如 D5XXX-YY-ZZZM1）

稳定性数据示例：
- 表格包含“0月”“3月”“6月”等列，并给出数值结果
- 多行检测项目（干燥失重、水分、杂质等）随时间的变化
- 记录药品在储存周期内质量变化的数据

只回答 "YES" 或 "NO"（不要输出其他文字）。
"""

    # NOTE: Added "even if cannot extract, output empty schema JSON" to avoid natural-language failures.
    EXTRACTION_PROMPT = """
你是“药品稳定性数据结构化抽取器（Stability Extractor）”。请从用户提供的文件内容中，抽取并结构化输出“产品稳定性考察信息与检测数据”。

# 重要：无论能否抽取到数据，都必须严格输出符合下方 JSON Schema 的一个 JSON object。
# 如果没有任何稳定性趋势数据：请输出空结构（batches 为空数组，unresolved_sections 可给原因），不要输出解释性文本。

# 目标
输出必须是【严格 JSON】（仅输出 JSON，不要输出任何解释、注释、Markdown、代码块或多余文字）。每份文件可能包含多个批次（Batch），必须全部识别并输出。

# 关键抽取字段（必须抽取）
1) 产品名称 product_name
2) 批次 batches[].batch_id
3) 标准/市场 regulatory_context（如：CEP、EDMF、USDMF、国内标准、国内拟申报等）
4) 考察条件 stability_condition（如：长期Ⅱ、长期ⅣB、加速、中间条件等）
5) 温湿度 temperature_c / humidity_rh（数值化；若范围则拆为 nominal + tolerance；无法拆分则用 raw）
6) 检测时间点 timepoints_months（如：0、3、6、9、12、18…；必须以“月”为单位的整数数组）
7) 每个检测时间点的“检测结果”以及“标准范围/限度”
   - 仅抽取【可定量/可比较】项目：例如 含量、有关物质/杂质（单杂/总杂）、水分、残留溶剂、粒度、比旋度、pH、重金属/元素杂质等
   - 必须同时抽取 spec（标准限度/范围）与 results（各时间点结果）

# 必须排除的项目（不要进入输出）
- 纯文本/定性项目：外观、鉴别、红外(IR)、颜色、气味、性状、结晶型“仅写符合/一致/相符/同图谱”等
- 若某项目虽然常见但该文件中只有“符合/Pass/Conforms”且没有可比较数值或阈值，也视为纯文本排除

# JSON 输出要求（严格）
- 只能输出一个顶层 JSON object
- 不允许 NaN/Infinity
- 所有键必须使用双引号
- 数值用 number；无法数值化的结果用 string
- 缺失信息用 null（不要省略 key）
- 同一批次可出现多个条件（如长期+加速），应拆分为多个 studies[] 记录

# 归一化与解析规则
A. 标准/市场 regulatory_context
- 从文件中的标题、页眉、备注、表头、文件名线索识别（例如“USDMF长期”“CEP”“EDMF”“国内拟申报”）
- 若出现多个，按最贴近该表/该段落的上下文分配；无法判断则放在 file_level.regulatory_context 并在 study 中继承

B. 考察条件 stability_condition 与温湿度
- condition_enum 仅允许：["long_term","intermediate","accelerated","stress","other"]
- 若文件写“长期Ⅱ/ⅣB”等：归入 long_term，并把原始写法写入 stability_condition_label
- 温湿度解析：
  - 能解析出“标称值±公差” -> temperature_c: {nominal: 25, tolerance: 2}；humidity_rh: {nominal: 60, tolerance: 5}
  - 若是范围（如 30-35℃） -> {min:30, max:35}
  - 若解析失败 -> raw 字段保留原文

C. 时间点
- 将“0月/初始/Initial”统一为 0
- 将“3M/3 months/3月”统一为 3
- 若表中缺某时间点结果，则该时间点仍可出现在 timepoints_months（若表头存在）；对应结果填 null

D. 项目与单位
- 为每个项目建立 analyte/item：
  - item_name 原文
  - normalized_name 尽量归一（如 “Assay (HPLC)” -> “assay”）
  - unit：从标准或结果中抽取（如 %、ppm、mg/g、µg/g、CFU/g 等）；缺失则 null
- 标准 spec：
  - type 仅允许：["range","max","min","equals","other"]
  - range -> {min, max}
  - max -> {max}
  - min -> {min}
  - equals -> {value}
  - 无法结构化 -> type="other", raw 保留

E. 结果 results_by_timepoint
- 对每个时间点输出一个对象：
  { "month": 3, "value": 99.8, "raw": "99.8", "qualifier": null }
- 若出现 "<LOD", "<LOQ", "ND"：
  - value: null
  - qualifier: "<LOD" / "<LOQ" / "ND"
  - raw 保留原文
  - 若同时给出 LOD/LOQ 数值，可填 detection_limit: {type:"LOD", value:0.01, unit:"%"} 等
- 若同一时间点出现重复测定（如两次结果）：
  - value: null
  - raw: "0.035 / 0.036"
  - replicate_values: [0.035, 0.036]（能解析就填）

F. 多批次与表结构
- 文件可能按批次分段，也可能一个表多批次并列。
- 必须识别每个 batch，并把对应数据放入各自的 batches[]。
- 若某段落/表无法明确属于哪个批次，放入 unresolved_sections[] 并给出原文片段（<=300字符）

# 置信度与可追溯性
- 每个 study 输出 confidence (0~1)
- 每个 item 输出 confidence (0~1)
- 记录 source_snippets：为每个 study 提供最多 3 条原文片段（每条<=200字符），用于人工核对

# 输出 JSON Schema（必须严格遵循）
{
  "file_level": {
    "product_name": null,
    "regulatory_context": null
  },
  "batches": [
    {
      "batch_id": null,
      "studies": [
        {
          "regulatory_context": null,
          "stability_condition": null,
          "stability_condition_label": null,
          "condition_enum": null,
          "temperature_c": {
            "nominal": null,
            "tolerance": null,
            "min": null,
            "max": null,
            "raw": null
          },
          "humidity_rh": {
            "nominal": null,
            "tolerance": null,
            "min": null,
            "max": null,
            "raw": null
          },
          "timepoints_months": [],
          "items": [
            {
              "item_name": null,
              "normalized_name": null,
              "unit": null,
              "spec": {
                "type": null,
                "min": null,
                "max": null,
                "value": null,
                "raw": null
              },
              "results_by_timepoint": [
                {
                  "month": null,
                  "value": null,
                  "raw": null,
                  "qualifier": null,
                  "replicate_values": null,
                  "detection_limit": {
                    "type": null,
                    "value": null,
                    "unit": null,
                    "raw": null
                  }
                }
              ],
              "confidence": null
            }
          ],
          "source_snippets": [],
          "confidence": null
        }
      ]
    }
  ],
  "unresolved_sections": [
    {
      "reason": null,
      "snippet": null
    }
  ]
}

# 输出约束
- 输出必须是有效 JSON
- 顶层字段必须完整出现：file_level, batches, unresolved_sections
- 禁止输出任何非 JSON 内容
"""

    # --- Heuristic patterns to reduce unnecessary LLM calls ---
    _RE_TIMEPOINT = re.compile(r"\b(0|3|6|9|12|18|24|36|48|60)\s*(月|months|month|M)\b", re.IGNORECASE)
    _RE_BATCH = re.compile(r"\b[A-Z]?\d{3,5}-\d{2}-\d{3,5}[A-Z0-9]{0,3}\b", re.IGNORECASE)
    _RE_NUM_ITEM_HINT = re.compile(r"(水分|含量|有关物质|杂质|总杂|单杂|残留溶剂|粒度|比旋度|pH|ppm|%|mg/g|ug/g|µg/g)", re.IGNORECASE)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self.model = model or LLM_MODEL

        if not self.api_key:
            raise ValueError("API key is required")

        # OpenAI compatible interface
        base_timeout = max(LLM_TIMEOUT_SCREENING, LLM_TIMEOUT_EXTRACTION)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=base_timeout)

    # ---------------------------
    # Low-level helpers
    # ---------------------------
    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        """Read file with encoding fallbacks."""
        if file_path.suffix.lower() == ".csv":
            encodings = ("utf-8-sig", "utf-8", "gb18030")
        else:
            encodings = ("utf-8", "utf-8-sig", "gb18030")

        for enc in encodings:
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue

        # last resort
        return file_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _parse_yes_no(text: str) -> Optional[bool]:
        """Parse YES/NO with tolerance to punctuation/whitespace."""
        if not text:
            return None
        s = text.strip().upper()
        if not s:
            return None
        first = s.split()[0].strip().strip(".,;:!\"'()[]{}")
        if first == "YES":
            return True
        if first == "NO":
            return False
        # fallback: contains check
        if "YES" in s and "NO" not in s:
            return True
        if "NO" in s and "YES" not in s:
            return False
        return None

    @staticmethod
    def _clean_code_fences(text: str) -> str:
        """Remove markdown code fences if present."""
        t = (text or "").strip()
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

    def _dump_invalid_json(self, raw_text: str, stage: str, source_name: Optional[str]) -> Path:
        """Write invalid JSON response to a text file for inspection."""
        invalid_dir = EXTRACTED_DIR / "invalid_json"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\-.]+", "_", source_name or "unknown")
        filename = f"{safe_name}_{stage}_{uuid.uuid4().hex[:8]}.txt"
        out_path = invalid_dir / filename
        out_path.write_text(raw_text or "", encoding="utf-8")
        return out_path

    def _call_llm(
        self,
        prompt: str,
        content: str,
        stage: str = "LLM",
        *,
        force_json_object: bool = False,
        temperature: Optional[float] = None,
        timeout_s: Optional[int] = None,
        source_name: Optional[str] = None,
    ) -> Any:
        """
        Call LLM with retry logic.
        If force_json_object=True, request strict JSON object via response_format.
        Returns:
            - if force_json_object: Python dict (parsed by SDK)
            - else: response text (str)
        """
        request_id = f"llm-{uuid.uuid4().hex[:10]}"
        total_attempts = MAX_RETRIES

        # Temperature strategy
        if temperature is None:
            if stage == "extraction":
                temperature = 0.0
            elif stage == "screening":
                temperature = min(float(LLM_TEMPERATURE), 0.3)
            else:
                temperature = float(LLM_TEMPERATURE)

        for attempt in range(MAX_RETRIES):
            stop_event = threading.Event()
            attempt_no = attempt + 1
            start_time = time.perf_counter()
            if timeout_s is None:
                timeout_s = LLM_TIMEOUT_EXTRACTION if stage.startswith("extraction") else LLM_TIMEOUT_SCREENING

            try:
                print(
                    f"[{request_id}] LLM调用开始 | 阶段={stage} | 尝试 {attempt_no}/{total_attempts} | "
                    f"model={self.model} | content_len={len(content)} | json={force_json_object} | timeout={timeout_s}s"
                )

                def _heartbeat() -> None:
                    while not stop_event.wait(10):
                        elapsed = time.perf_counter() - start_time
                        print(f"[{request_id}] LLM等待中 | 阶段={stage} | 已等待 {elapsed:.1f}s")

                heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
                heartbeat_thread.start()

                def _request():
                    kwargs = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": content},
                        ],
                        "temperature": temperature,
                        "max_tokens": LLM_MAX_TOKENS,
                        "timeout": timeout_s,
                    }
                    if force_json_object:
                        kwargs["response_format"] = {"type": "json_object"}
                    return self.client.chat.completions.create(**kwargs)

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(_request)
                timed_out = False
                try:
                    response = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError as e:
                    timed_out = True
                    future.cancel()
                    stop_event.set()
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[{request_id}] LLM调用超时 | 阶段={stage} | 尝试 {attempt_no}/{total_attempts} | "
                        f"用时 {elapsed:.2f}s | timeout={timeout_s}s"
                    )
                    # Don't wait for the hung worker thread; it can block shutdown.
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    raise Exception(f"LLM API call timed out after {timeout_s}s") from e
                finally:
                    # If the request finished, shut down cleanly; if not, avoid blocking.
                    executor.shutdown(wait=not timed_out, cancel_futures=True)

                stop_event.set()
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{request_id}] LLM调用结束 | 阶段={stage} | 尝试 {attempt_no}/{total_attempts} | 用时 {elapsed:.2f}s"
                )

                content_out = response.choices[0].message.content
                if force_json_object:
                    cleaned = self._clean_code_fences(content_out)
                    try:
                        return json.loads(cleaned)
                    except Exception as parse_err:
                        out_path = self._dump_invalid_json(content_out or "", stage=stage, source_name=source_name)
                        print(f"[{request_id}] 非法JSON已导出 | 阶段={stage} | 路径={out_path}")
                        raise parse_err
                else:
                    return (content_out or "").strip()

            except Exception as e:
                stop_event.set()
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{request_id}] LLM调用失败 | 阶段={stage} | 尝试 {attempt_no}/{total_attempts} | "
                    f"用时 {elapsed:.2f}s | 错误: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise Exception(f"LLM API call failed after {MAX_RETRIES} attempts: {e}") from e

        # unreachable
        raise Exception("LLM call failed unexpectedly")

    # ---------------------------
    # Validation & normalization
    # ---------------------------
    def _normalize_extracted_data(self, data: Any) -> List[Dict]:
        if data is None:
            return []
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def _is_valid_extracted(self, data: Dict) -> bool:
        """Stronger schema check than original."""
        if not isinstance(data, dict):
            return False
        if not all(k in data for k in ("file_level", "batches", "unresolved_sections")):
            return False
        if not isinstance(data.get("file_level"), dict):
            return False
        if not isinstance(data.get("batches"), list):
            return False
        if not isinstance(data.get("unresolved_sections"), list):
            return False

        for b in data["batches"]:
            if not isinstance(b, dict):
                return False
            if "batch_id" not in b or "studies" not in b:
                return False
            if not isinstance(b.get("studies"), list):
                return False

        return True

    # ---------------------------
    # Heuristic screening
    # ---------------------------
    def _heuristic_has_stability_signals(self, text: str) -> bool:
        """Cheap pre-check: if strong signals exist, skip LLM screening."""
        if not text:
            return False
        has_tp = bool(self._RE_TIMEPOINT.search(text))
        has_batch = bool(self._RE_BATCH.search(text))
        has_num_item = bool(self._RE_NUM_ITEM_HINT.search(text))
        # Require timepoints + (batch or numeric item hint)
        return has_tp and (has_batch or has_num_item)

    def is_stability_data(self, file_content: str) -> bool:
        """Check if file contains stability trend data."""
        # If heuristics indicate strong presence, skip LLM screening
        if self._heuristic_has_stability_signals(file_content):
            return True

        try:
            response = self._call_llm(self.SCREENING_PROMPT, file_content, stage="screening", force_json_object=False)
            parsed = self._parse_yes_no(response)
            return bool(parsed) if parsed is not None else False
        except Exception as e:
            print(f"Error screening file: {e}")
            return False

    # ---------------------------
    # Chunking & merging
    # ---------------------------
    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 80000, overlap: int = 2000) -> List[str]:
        """Split long text into overlapping chunks to avoid losing tail content."""
        if not text:
            return [""]
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        i = 0
        n = len(text)
        while i < n:
            j = min(i + chunk_size, n)
            chunks.append(text[i:j])
            if j >= n:
                break
            i = max(0, j - overlap)
        return chunks

    @staticmethod
    def _study_key(study: Dict) -> Tuple:
        """Key to de-duplicate/merge studies."""
        if not isinstance(study, dict):
            return ("", "", "", "", "", "")
        return (
            study.get("regulatory_context"),
            study.get("stability_condition"),
            study.get("stability_condition_label"),
            json.dumps(study.get("temperature_c", {}), ensure_ascii=False, sort_keys=True),
            json.dumps(study.get("humidity_rh", {}), ensure_ascii=False, sort_keys=True),
            study.get("condition_enum"),
        )

    @staticmethod
    def _item_key(item: Dict) -> Tuple:
        """Key to de-duplicate/merge items."""
        if not isinstance(item, dict):
            return ("", "", "", "", "")
        spec = item.get("spec", {})
        return (
            item.get("normalized_name") or "",
            item.get("item_name") or "",
            item.get("unit") or "",
            (spec.get("type") if isinstance(spec, dict) else None),
            (spec.get("raw") if isinstance(spec, dict) else None),
        )

    @staticmethod
    def _merge_timepoint_results(existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
        """Merge results_by_timepoint by month, preferring non-null values."""
        by_month: Dict[int, Dict] = {}
        for r in existing or []:
            if isinstance(r, dict) and isinstance(r.get("month"), int):
                by_month[r["month"]] = r

        for r in incoming or []:
            if not (isinstance(r, dict) and isinstance(r.get("month"), int)):
                continue
            m = r["month"]
            if m not in by_month:
                by_month[m] = r
                continue
            # merge fields: prefer non-null in existing; fill gaps from incoming
            cur = by_month[m]
            merged = dict(cur)
            for k, v in r.items():
                if merged.get(k) is None and v is not None:
                    merged[k] = v
            # detection_limit nested merge
            if isinstance(cur.get("detection_limit"), dict) and isinstance(r.get("detection_limit"), dict):
                dl = dict(cur["detection_limit"])
                for k, v in r["detection_limit"].items():
                    if dl.get(k) is None and v is not None:
                        dl[k] = v
                merged["detection_limit"] = dl
            by_month[m] = merged

        # return sorted by month
        return [by_month[m] for m in sorted(by_month.keys())]

    def _merge_extraction_dicts(self, base: Dict, add: Dict) -> Dict:
        """Merge two extraction outputs that follow the schema."""
        if not self._is_valid_extracted(base):
            return add
        if not self._is_valid_extracted(add):
            return base

        merged = dict(base)

        # file_level merge (prefer non-null)
        fl = dict(merged.get("file_level", {}))
        afl = add.get("file_level", {}) if isinstance(add.get("file_level"), dict) else {}
        for k in ("product_name", "regulatory_context"):
            if fl.get(k) is None and afl.get(k) is not None:
                fl[k] = afl.get(k)
        merged["file_level"] = fl

        # unresolved_sections concat (keep short)
        merged_unres = list(merged.get("unresolved_sections") or [])
        add_unres = list(add.get("unresolved_sections") or [])
        merged_unres.extend([u for u in add_unres if isinstance(u, dict)])
        merged["unresolved_sections"] = merged_unres[:200]  # hard cap

        # batches merge by batch_id
        base_batches = merged.get("batches") or []
        add_batches = add.get("batches") or []
        batch_map: Dict[str, Dict] = {}

        def _batch_id(b: Dict) -> str:
            bid = b.get("batch_id")
            return str(bid).strip() if bid is not None else ""

        for b in base_batches:
            if not isinstance(b, dict):
                continue
            bid = _batch_id(b)
            batch_map[bid] = b

        for b in add_batches:
            if not isinstance(b, dict):
                continue
            bid = _batch_id(b)
            if bid not in batch_map:
                batch_map[bid] = b
                continue

            # merge studies
            existing_batch = batch_map[bid]
            ex_studies = existing_batch.get("studies") or []
            in_studies = b.get("studies") or []

            ex_map: Dict[Tuple, Dict] = {}
            for st in ex_studies:
                if isinstance(st, dict):
                    ex_map[self._study_key(st)] = st

            for st in in_studies:
                if not isinstance(st, dict):
                    continue
                key = self._study_key(st)
                if key not in ex_map:
                    ex_map[key] = st
                    continue

                # merge study fields
                cur = ex_map[key]
                merged_st = dict(cur)

                # fill simple fields
                for k2, v2 in st.items():
                    if k2 in ("items", "timepoints_months", "source_snippets"):
                        continue
                    if merged_st.get(k2) is None and v2 is not None:
                        merged_st[k2] = v2

                # timepoints_months union
                tp1 = merged_st.get("timepoints_months") or []
                tp2 = st.get("timepoints_months") or []
                if isinstance(tp1, list) and isinstance(tp2, list):
                    tp_union = sorted({t for t in tp1 + tp2 if isinstance(t, int)})
                    merged_st["timepoints_months"] = tp_union

                # source_snippets concat
                ss1 = merged_st.get("source_snippets") or []
                ss2 = st.get("source_snippets") or []
                if isinstance(ss1, list) and isinstance(ss2, list):
                    merged_st["source_snippets"] = (ss1 + ss2)[:6]

                # items merge by key
                items1 = merged_st.get("items") or []
                items2 = st.get("items") or []
                item_map: Dict[Tuple, Dict] = {}
                for it in items1:
                    if isinstance(it, dict):
                        item_map[self._item_key(it)] = it
                for it in items2:
                    if not isinstance(it, dict):
                        continue
                    ik = self._item_key(it)
                    if ik not in item_map:
                        item_map[ik] = it
                    else:
                        cur_it = item_map[ik]
                        new_it = dict(cur_it)
                        # fill fields
                        for k3, v3 in it.items():
                            if k3 == "results_by_timepoint":
                                continue
                            if new_it.get(k3) is None and v3 is not None:
                                new_it[k3] = v3
                        # merge results_by_timepoint
                        new_it["results_by_timepoint"] = self._merge_timepoint_results(
                            cur_it.get("results_by_timepoint") or [],
                            it.get("results_by_timepoint") or [],
                        )
                        item_map[ik] = new_it

                merged_st["items"] = list(item_map.values())
                ex_map[key] = merged_st

            existing_batch["studies"] = list(ex_map.values())
            batch_map[bid] = existing_batch

        merged["batches"] = list(batch_map.values())
        return merged

    # ---------------------------
    # Extraction
    # ---------------------------
    def extract_stability_data(self, file_content: str, source_name: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Extract structured stability data from file content.

        For large docs, run chunked extraction then merge to avoid losing tail data.
        """
        try:
            # Chunking to avoid truncation. Tune chunk_size if needed.
            chunks = self._split_into_chunks(file_content, chunk_size=80000, overlap=2000)

            merged: Optional[Dict] = None
            for idx, chunk in enumerate(chunks, start=1):
                stage = "extraction" if len(chunks) == 1 else f"extraction_chunk_{idx}/{len(chunks)}"
                data = self._call_llm(
                    self.EXTRACTION_PROMPT,
                    chunk,
                    stage="extraction",
                    force_json_object=True,
                    temperature=0.0,
                    source_name=source_name,
                )

                # Normalize + validate
                items = self._normalize_extracted_data(data)
                items = [item for item in items if self._is_valid_extracted(item)]

                if not items:
                    continue

                # Only one object is expected, but we support multiple defensively
                for obj in items:
                    if merged is None:
                        merged = obj
                    else:
                        merged = self._merge_extraction_dicts(merged, obj)

            if merged is None:
                return None

            return [merged]

        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

    # ---------------------------
    # File-level analysis
    # ---------------------------
    def analyze_file(self, file_path: Path) -> Optional[List[Dict]]:
        """
        Analyze a single file and extract stability data.
        Returns a list of extracted dicts (usually length=1), or None if no stability data.
        """
        try:
            content = self._read_text_file(file_path)

            if not self.is_stability_data(content):
                return None

            items = self.extract_stability_data(content, source_name=file_path.name)
            if not items:
                return None

            # Add metadata
            for item in items:
                item["_metadata"] = {
                    "source_file": str(file_path),
                    "file_type": file_path.suffix.lower(),
                }

            return items

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None

    def batch_analyze_files(self, file_paths: List[Path]) -> List[Dict]:
        """Analyze multiple files in batch."""
        results: List[Dict] = []
        for file_path in file_paths:
            print(f"Analyzing: {file_path.name}")
            result = self.analyze_file(file_path)
            if result:
                results.extend(result)

                if len(result) == 1 and isinstance(result[0], dict):
                    item = result[0]
                    file_level = item.get("file_level", {}) if isinstance(item.get("file_level"), dict) else {}
                    product = file_level.get("product_name", "Unknown")
                    batch_count = len(item.get("batches", [])) if isinstance(item.get("batches"), list) else 0
                    print(f"  ✓ Extracted: {product} - {batch_count} batch(es)")
                else:
                    print(f"  ✓ Extracted {len(result)} records")
            else:
                print(f"  ✗ No stability data found")

        return results
