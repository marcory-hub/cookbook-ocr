"""
Surya OCR engine wrapper.

Surya is a transformer-based document OCR (https://github.com/VikParuchuri/surya).
It performs text detection + recognition + reading-order analysis in one pass,
and emits structured output including bold/italic/math tags — significantly
more accurate than Tesseract for multi-column book layouts with illustrations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from wordfreq import zipf_frequency

log = logging.getLogger(__name__)

_MATH_FRAC = re.compile(r"\\frac\{(\d+)\}\{(\d+)\}")
_FRAC_UNICODE = {
    (1, 2): "½", (1, 3): "⅓", (2, 3): "⅔",
    (1, 4): "¼", (3, 4): "¾",
    (1, 5): "⅕", (2, 5): "⅖", (3, 5): "⅗", (4, 5): "⅘",
    (1, 6): "⅙", (5, 6): "⅚",
    (1, 7): "⅐",
    (1, 8): "⅛", (3, 8): "⅜", (5, 8): "⅝", (7, 8): "⅞",
    (1, 9): "⅑", (1, 10): "⅒",
}
_MATH_TAG = re.compile(r"<math>(.*?)</math>", re.DOTALL)
_BOLD_TAG = re.compile(r"<b>(.*?)</b>", re.DOTALL)
_ITALIC_TAG = re.compile(r"<i>(.*?)</i>", re.DOTALL)
_OTHER_TAGS = re.compile(r"</?[a-zA-Z][^>]*>")


@dataclass
class SuryaLine:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float


def _html_to_plain(html: str) -> str:
    """Extract plain text from Surya block HTML (0.20+)."""
    if not html or not html.strip():
        return ""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    except Exception:
        return _OTHER_TAGS.sub("", html).strip()


class SuryaEngine:
    """Lazy-loaded Surya predictors shared across pages."""

    def __init__(self) -> None:
        self._recognition = None
        self._detection = None
        self._api: str = ""  # "v017" torch/MPS | "v020" VLM/llamacpp

    def _load(self) -> None:
        if self._recognition is not None:
            return
        log.info("Loading Surya models (first run downloads weights on first use)...")
        try:
            from surya.foundation import FoundationPredictor
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            foundation = FoundationPredictor()
            self._recognition = RecognitionPredictor(foundation)
            self._detection = DetectionPredictor()
            self._api = "v017"
        except ImportError:
            from surya.recognition import RecognitionPredictor

            self._recognition = RecognitionPredictor()
            self._api = "v020"
        log.info("Surya models ready (%s).", self._api)

    def ocr_page(self, bgr: np.ndarray) -> tuple[list[SuryaLine], float]:
        """OCR a BGR numpy image. Returns (lines_in_reading_order, mean_conf)."""
        self._load()
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            rgb = bgr[:, :, ::-1]
        else:
            rgb = bgr
        pil = Image.fromarray(rgb)

        if self._api == "v020":
            preds = self._recognition([pil], full_page=True)
            result = preds[0]
            lines: list[SuryaLine] = []
            confs: list[float] = []
            for blk in sorted(result.blocks, key=lambda b: b.reading_order):
                if blk.skipped or blk.error:
                    continue
                text = _clean_surya_text(_html_to_plain(blk.html))
                if not text.strip():
                    continue
                bbox = tuple(blk.bbox)
                conf = float(blk.confidence or 0.0)
                lines.append(SuryaLine(text=text, bbox=bbox, confidence=conf))
                confs.append(conf)
            mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
            return lines, mean_conf

        preds = self._recognition(
            [pil], det_predictor=self._detection, sort_lines=True
        )
        result = preds[0]
        lines = []
        confs = []
        for ln in result.text_lines:
            text = _clean_surya_text(ln.text)
            if not text.strip():
                continue
            bbox = tuple(ln.bbox)
            lines.append(SuryaLine(text=text, bbox=bbox, confidence=float(ln.confidence or 0)))
            confs.append(float(ln.confidence or 0))
        mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
        return lines, mean_conf

    def ocr_pages(
        self,
        bgr_images: list[np.ndarray],
        *,
        batch_size: int = 2,
    ) -> list[tuple[list[SuryaLine], float]]:
        """OCR multiple pages; batch through Surya where the API allows."""
        if not bgr_images:
            return []
        if batch_size < 1:
            batch_size = 1
        out: list[tuple[list[SuryaLine], float]] = []
        for start in range(0, len(bgr_images), batch_size):
            chunk = bgr_images[start : start + batch_size]
            if batch_size == 1 or self._api == "v020":
                for bgr in chunk:
                    out.append(self.ocr_page(bgr))
            else:
                out.extend(self._ocr_page_batch_v017(chunk))
            self.clear_mps_cache()
        return out

    def _ocr_page_batch_v017(
        self, bgr_images: list[np.ndarray]
    ) -> list[tuple[list[SuryaLine], float]]:
        self._load()
        pils: list[Image.Image] = []
        for bgr in bgr_images:
            rgb = bgr[:, :, ::-1] if bgr.ndim == 3 and bgr.shape[2] == 3 else bgr
            pils.append(Image.fromarray(rgb))
        preds = self._recognition(
            pils, det_predictor=self._detection, sort_lines=True
        )
        results: list[tuple[list[SuryaLine], float]] = []
        for result in preds:
            lines: list[SuryaLine] = []
            confs: list[float] = []
            for ln in result.text_lines:
                text = _clean_surya_text(ln.text)
                if not text.strip():
                    continue
                lines.append(
                    SuryaLine(
                        text=text,
                        bbox=tuple(ln.bbox),
                        confidence=float(ln.confidence or 0),
                    )
                )
                confs.append(float(ln.confidence or 0))
            mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
            results.append((lines, mean_conf))
        return results

    @staticmethod
    def clear_mps_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _math_to_unicode(match: re.Match) -> str:
    """Convert LaTeX math like `\\frac{1}{2}` to Unicode fractions where possible."""
    body = match.group(1)

    def _frac_repl(m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        return _FRAC_UNICODE.get((num, den), f"{num}/{den}")

    body = _MATH_FRAC.sub(_frac_repl, body)
    # Strip any remaining LaTeX control sequences defensively.
    body = re.sub(r"\\[a-zA-Z]+\s*", "", body)
    body = body.replace("{", "").replace("}", "")
    return body


def _clean_surya_text(text: str) -> str:
    """Convert Surya's HTML-ish + LaTeX output to plain markdown-safe text."""
    text = _MATH_TAG.sub(_math_to_unicode, text)
    text = _BOLD_TAG.sub(r"**\1**", text)
    text = _ITALIC_TAG.sub(r"*\1*", text)
    text = _OTHER_TAGS.sub("", text)
    text = _normalize_inline_fractions(text)
    # Unicode bullet glyphs the publisher used for tip lists become markdown "- ".
    text = re.sub(r"^\s*[•●◆▪][\s]+", "- ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# Surya often emits mixed-number fractions using the Unicode fraction slash
# (U+2044): "1½" becomes "11⁄2", "2¾" becomes "23⁄4". Also occasionally uses
# ASCII slash: "1 1/2 cups", "23/4 cups".
_FRAC_SLASH = "\u2044"
_UNIT_KW = r"cup|cups|tsp|tbsp|teaspoon|teaspoons|tablespoon|tablespoons|pound|pounds|oz|lb|lbs|quart|pint|gallon|stick|sticks"
_MIXED_FRAC_SLASH = re.compile(rf"(\d)(\d)[{_FRAC_SLASH}/](\d)")
_BARE_FRAC_SLASH = re.compile(rf"(?<!\d)(\d)[{_FRAC_SLASH}](\d)(?!\d)")


def _mixed_frac_repl(m: re.Match) -> str:
    whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = _FRAC_UNICODE.get((num, den))
    return f"{whole}{frac}" if frac else m.group(0)


def _bare_frac_repl(m: re.Match) -> str:
    num, den = int(m.group(1)), int(m.group(2))
    return _FRAC_UNICODE.get((num, den), m.group(0))


def _normalize_inline_fractions(text: str) -> str:
    text = _MIXED_FRAC_SLASH.sub(_mixed_frac_repl, text)
    text = _BARE_FRAC_SLASH.sub(_bare_frac_repl, text)
    # ASCII "1 1/2 cup" / "2 3/4 cups" — require unit context for safety.
    for num, den, uni in [
        (1, 2, "½"), (1, 3, "⅓"), (2, 3, "⅔"),
        (1, 4, "¼"), (3, 4, "¾"),
        (1, 8, "⅛"), (3, 8, "⅜"), (5, 8, "⅝"), (7, 8, "⅞"),
    ]:
        pat = re.compile(rf"(\d)\s+{num}/{den}(?=\s+(?:{_UNIT_KW})\b)", re.I)
        text = pat.sub(rf"\1{uni}", text)
        pat2 = re.compile(rf"(?<!\d){num}/{den}(?=\s+(?:{_UNIT_KW})\b)", re.I)
        text = pat2.sub(uni, text)
    return text


def assemble_markdown(lines: list[SuryaLine], page_width: int) -> str:
    """
    Reassemble Surya's text lines into a reading-order markdown body.

    Workflow:
      1. Cluster all lines into columns by x-start.
      2. Single column → emit linearly by y.
      3. Multi-column → pull full-width "banner" lines out as their own
         one-line sections so they interleave by y-position, and emit each
         column as its own block.
    """
    if not lines:
        return ""

    median_h = float(np.median([ln.bbox[3] - ln.bbox[1] for ln in lines]))
    paragraph_gap = max(8.0, median_h * 1.4)

    columns = _cluster_columns(lines, page_width)
    if len(columns) == 1:
        return _emit_linear(columns[0], paragraph_gap)

    # Multi-column page: pull banners out so they interleave correctly.
    banners: list[SuryaLine] = []
    per_col: list[list[SuryaLine]] = [[] for _ in columns]
    for i, col in enumerate(columns):
        for ln in col:
            x0, _, x1, _ = ln.bbox
            if x1 - x0 > page_width * 0.58:
                banners.append(ln)
            else:
                per_col[i].append(ln)

    sections: list[list[SuryaLine]] = []
    for b in banners:
        sections.append([b])
    for col in per_col:
        if col:
            sections.append(col)

    for s in sections:
        s.sort(key=lambda ln: ln.bbox[1])
    sections.sort(key=lambda s: min(ln.bbox[1] for ln in s))

    out: list[str] = []
    for section in sections:
        if out and out[-1] != "":
            out.append("")
        out.append(_emit_linear(section, paragraph_gap))

    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out).strip()


def _emit_linear(lines: list[SuryaLine], paragraph_gap: float) -> str:
    """Emit lines grouped into markdown blocks: headings, lists, paragraphs."""
    if not lines:
        return ""
    sorted_lines = sorted(lines, key=lambda l: l.bbox[1])

    # Font-height-based heading hint: wrap visually-larger isolated lines in
    # bold so the downstream formatter turns them into ### headings.
    heights = [ln.bbox[3] - ln.bbox[1] for ln in sorted_lines]
    median_h = float(np.median(heights)) if heights else 0.0

    raw_blocks: list[list[str]] = []
    current: list[str] = []
    prev_y_bottom: Optional[float] = None
    for ln in sorted_lines:
        text = ln.text.strip()
        if not text:
            continue
        height = ln.bbox[3] - ln.bbox[1]
        if _is_font_heading(text, height, median_h):
            text = f"**{text}**"

        is_bold_line = bool(_ALL_BOLD_LINE.match(text))
        prev_is_bold = bool(current and _ALL_BOLD_LINE.match(current[-1]))
        ygap = (ln.bbox[1] - prev_y_bottom) if prev_y_bottom is not None else 0.0
        force_break = is_bold_line or prev_is_bold
        ygap_break = prev_y_bottom is not None and ygap > paragraph_gap

        if (force_break or ygap_break) and current:
            raw_blocks.append(current)
            current = []
        current.append(text)
        prev_y_bottom = ln.bbox[3]
    if current:
        raw_blocks.append(current)

    rendered: list[str] = []
    for block in raw_blocks:
        md = _format_block(block)
        if md:
            rendered.append(md)
    return "\n\n".join(rendered)


# --- markdown block formatting -----------------------------------------------

_FRACTION_CHARS = set("¼½¾⅓⅔⅛⅜⅝⅞⅕⅖⅗⅘⅙⅚⅐⅑⅒")
_ALL_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*$")
_QUANTITY_START = re.compile(
    r"^\s*(?:"
    r"[¼½¾⅓⅔⅛⅜⅝⅞⅕⅖⅗⅘⅙⅚⅐⅑⅒]"           # bare vulgar fraction
    r"|\d+(?:[¼½¾⅓⅔⅛⅜⅝⅞]|\s*[-–]\s*[¼½¾⅓⅔⅛⅜⅝⅞]|/\d+|\.\d+)?\s"
    r")"
)
_BARE_FRACTION_RANGE = re.compile(r"^[¼½¾⅓⅔⅛⅜⅝⅞]\s*[-–]\s*[¼½¾⅓⅔⅛⅜⅝⅞]\s")
_SEASONING_PREFIX = re.compile(
    r"^(?:Salt|Pepper|Cayenne|Paprika|Sugar|Flour|Oil|Butter|Water|Milk|Cream|"
    r"Egg|Eggs|Vinegar|Lemon|Lime|Wine|Stock|Broth|Yeast)s?\b",
    re.I,
)


def _format_block(lines: list[str]) -> str:
    """Format a block of raw line texts as a markdown chunk."""
    if not lines:
        return ""

    if len(lines) == 1:
        line = lines[0]
        bold_match = _ALL_BOLD_LINE.match(line)
        if bold_match:
            return f"### {bold_match.group(1).strip()}"
        return line

    # Multi-line block: route through the mixed-mode formatter which detects
    # ingredient lists, prose paragraphs, and their transitions within a block.
    return _format_mixed(lines)


def _format_mixed(lines: list[str]) -> str:
    """
    Render a block, interleaving headings, lists, and prose paragraphs.

    A line is treated as a list item only when there's clear context:
      1. The previous line ended with ":" (setup for a list), or
      2. Another quantity-starting line follows, or
      3. We're already inside a list (continuation).

    Lines that happen to start with a digit mid-prose ("12 minutes. Add ...")
    stay part of the prose paragraph.
    """
    out: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            if out and out[-1] != "":
                out.append("")
            out.append(_smart_join(paragraph))
            out.append("")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            if out and out[-1] != "":
                out.append("")
            out.extend(f"- {item}" for item in list_items)
            out.append("")
            list_items.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        bold_match = _ALL_BOLD_LINE.match(line)
        if bold_match:
            flush_list()
            flush_paragraph()
            out.append(f"### {bold_match.group(1).strip()}")
            out.append("")
            i += 1
            continue

        # Pre-bulleted line (publisher tip bullet "•" normalized to "- ").
        pre_bulleted = line.startswith("- ")
        is_quantity = _starts_with_quantity(line) or _looks_like_seasoning_item(line)

        # Continuation of an existing list item?
        if list_items and not is_quantity and not pre_bulleted and _is_item_continuation(list_items[-1], line):
            list_items[-1] = _join_with_hyphen(list_items[-1], line)
            i += 1
            continue

        if pre_bulleted:
            flush_paragraph()
            list_items.append(line[2:].strip())
            i += 1
            continue

        if is_quantity:
            # Continue an in-progress list.
            if list_items:
                list_items.append(line)
                i += 1
                continue
            # Start a new list only with explicit context.
            prev_ends_colon = bool(
                paragraph and paragraph[-1].rstrip().endswith(":")
            ) or (
                out and out[-1] == "" and len(out) >= 2 and out[-2].rstrip().endswith(":")
            )
            nxt_is_quantity = i + 1 < n and (
                _starts_with_quantity(lines[i + 1]) or _looks_like_seasoning_item(lines[i + 1])
            )
            if prev_ends_colon or nxt_is_quantity:
                flush_paragraph()
                list_items.append(line)
                i += 1
                continue
            # Otherwise: prose line that happens to start with a number.

        # Not a list item → prose. Close any open list first.
        flush_list()

        if paragraph:
            prev = paragraph[-1].rstrip()
            starts_upper = line.lstrip()[:1].isupper() if line.lstrip() else False
            if prev and prev[-1] in ".!?" and starts_upper and len(prev) > 20:
                flush_paragraph()
        paragraph.append(line)
        i += 1

    flush_list()
    flush_paragraph()

    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _starts_with_quantity(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    if s[0] in _FRACTION_CHARS:
        return True
    return bool(_QUANTITY_START.match(s) or _BARE_FRACTION_RANGE.match(s))


def _looks_like_seasoning_item(line: str) -> bool:
    """Short ingredient-style lines without quantities ("Cayenne pepper")."""
    return bool(_SEASONING_PREFIX.match(line.strip())) and len(line) < 40


def _format_list(lines: list[str]) -> str:
    items: list[str] = []
    preamble: Optional[str] = None
    trailing_prose: list[str] = []
    list_ended = False

    for i, line in enumerate(lines):
        if list_ended:
            trailing_prose.append(line)
            continue
        if i == 0 and line.rstrip().endswith(":") and not _starts_with_quantity(line):
            preamble = line
            continue
        if _starts_with_quantity(line) or _looks_like_seasoning_item(line):
            items.append(line)
        elif items and _is_item_continuation(items[-1], line):
            items[-1] = _join_with_hyphen(items[-1], line)
        elif items:
            # Looks like prose — the list is over.
            list_ended = True
            trailing_prose.append(line)
        else:
            preamble = f"{preamble} {line.strip()}" if preamble else line

    out: list[str] = []
    if preamble:
        out.append(preamble)
        out.append("")
    out.extend(f"- {item}" for item in items)
    if trailing_prose:
        out.append("")
        out.append(_join_prose(trailing_prose))
    return "\n".join(out)


def _is_item_continuation(prev_item: str, curr_line: str) -> bool:
    """True if curr_line is a wrapped continuation of prev_item."""
    curr = curr_line.strip()
    if not curr:
        return False
    # Parenthetical aside, conjunction, or lowercase continuation.
    if curr.startswith("("):
        return True
    if curr[:1].islower():
        return True
    # Short phrase that reads as a continuation (e.g., "and cut into...").
    if len(curr) < 40 and not _looks_sentence_like(curr):
        return True
    return False


def _looks_sentence_like(text: str) -> bool:
    """Rough check: starts with an imperative verb or capitalized sentence."""
    if not text or not text[0].isupper():
        return False
    # Multiple tokens and ends with terminal punctuation.
    if text.rstrip()[-1:] in ".!?":
        return len(text.split()) >= 3
    return False


_LEFT_HYPHEN_WORD = re.compile(r"(\w+)-$")
_RIGHT_HEAD_WORD = re.compile(r"^(\w+)")


def _is_font_heading(text: str, height: float, median_h: float) -> bool:
    """Should a visually-larger line be promoted to a heading?"""
    if not text or median_h <= 0:
        return False
    if text.startswith("**"):
        return False  # already bold-wrapped
    if height <= median_h * 1.35:
        return False
    if len(text) >= 60:
        return False
    if not any(ch.islower() for ch in text):
        return False  # all-caps fragment (comic SFX, data cell)
    if text[:1].isdigit():
        return False  # data row ("30 ML NASAL PASSAGES")
    if "http" in text.lower() or "www." in text.lower():
        return False
    last = text.rstrip()[-1] if text.rstrip() else ""
    if last in ".!,);:":
        return False  # prose tail / data cell endings
    return True


def _join_with_hyphen(left: str, right: str) -> str:
    """
    Join two lines where the left ends in a hyphen.

    Disambiguate line-wrap hyphens ("table-\\nspoons" → "tablespoons") from
    compound-word hyphens ("medium-\\nlow" → "medium-low") by checking which
    form is more common in English via wordfreq.
    """
    left = left.rstrip()
    right = right.strip()
    if not (left.endswith("-") and right[:1].islower()):
        return f"{left} {right}" if right else left

    lm = _LEFT_HYPHEN_WORD.search(left)
    rm = _RIGHT_HEAD_WORD.match(right)
    if lm and rm:
        merged = (lm.group(1) + rm.group(1)).lower()
        hyphenated = (lm.group(1) + "-" + rm.group(1)).lower()
        merged_freq = zipf_frequency(merged, "en")
        hyphen_freq = zipf_frequency(hyphenated, "en")
        # Merged wins unless it's clearly not a real word and the hyphenated
        # form is at least as common as the merged form.
        if merged_freq < 2.0 and hyphen_freq >= merged_freq:
            return f"{left}{right}"  # keep hyphen
    return left[:-1] + right


def _join_prose(lines: list[str]) -> str:
    """Join wrapped prose lines into flowing sentences."""
    if not lines:
        return ""
    # Split internal into paragraphs on terminal punctuation + next-line capitalization.
    paragraphs: list[list[str]] = [[lines[0]]]
    for line in lines[1:]:
        prev = paragraphs[-1][-1].rstrip()
        starts_upper = line.lstrip()[:1].isupper() if line.lstrip() else False
        # Start a new paragraph when previous line ended a sentence AND this line
        # begins a new capitalized sentence (standard prose break).
        if prev and prev[-1] in ".!?" and starts_upper and len(prev) > 20:
            paragraphs.append([line])
        else:
            paragraphs[-1].append(line)

    out_paragraphs: list[str] = []
    for p in paragraphs:
        out_paragraphs.append(_smart_join(p))
    return "\n\n".join(out_paragraphs)


def _smart_join(lines: list[str]) -> str:
    """Join lines, handling hyphenated line-break wraps."""
    if not lines:
        return ""
    result = lines[0].strip()
    for line in lines[1:]:
        result = _join_with_hyphen(result, line)
    return result


def _cluster_columns(lines: list[SuryaLine], page_width: int) -> list[list[SuryaLine]]:
    """
    Cluster text lines into columns by their left-edge x-coordinate.

    Returns a single-column result when splits would produce a minor cluster
    that's likely just an indented paragraph or decorative sidebar rather than
    a true second column.
    """
    if not lines:
        return []
    single = [sorted(lines, key=lambda l: l.bbox[1])]
    if len(lines) < 4:
        return single

    x0s = sorted(ln.bbox[0] for ln in lines)
    gap_threshold = page_width * 0.12  # only real column gutters cross this
    cluster_starts = [x0s[0]]
    for prev, curr in zip(x0s, x0s[1:]):
        if curr - prev > gap_threshold:
            cluster_starts.append(curr)
    if len(cluster_starts) == 1:
        return single

    centers = cluster_starts
    clusters: list[list[SuryaLine]] = [[] for _ in centers]
    for ln in lines:
        x0 = ln.bbox[0]
        idx = min(range(len(centers)), key=lambda i: abs(x0 - centers[i]))
        clusters[idx].append(ln)
    clusters = [c for c in clusters if c]

    # Reject the split if any minor cluster is too small — it's probably a
    # caption, pull-quote, or indented block inside a single-column layout.
    total = sum(len(c) for c in clusters)
    min_share = 0.2
    min_count = 4
    if any(len(c) < max(min_count, total * min_share) for c in clusters):
        return single

    clusters.sort(key=lambda c: min(ln.bbox[0] for ln in c))
    return clusters
