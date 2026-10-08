"""
Surya layout-aware OCR engine.

Uses Surya LayoutPredictor for reading order, header/footer removal, and
figure/caption placement, and Surya recognition for text. Lines are assigned
to the layout block whose bbox contains the line center.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from surya_engine import (
    SuryaEngine,
    SuryaLine,
    _clean_surya_text,
    _join_prose,
    _smart_join,
)

log = logging.getLogger(__name__)

# Labels from Surya LayoutPredictor (0.17.x).
LABEL_PAGE_HEADER = "PageHeader"
LABEL_PAGE_FOOTER = "PageFooter"
LABEL_SECTION_HEADER = "SectionHeader"
LABEL_TEXT = "Text"
LABEL_LIST_ITEM = "ListItem"
LABEL_TABLE = "Table"
LABEL_PICTURE = "Picture"
LABEL_FIGURE = "Figure"
LABEL_CAPTION = "Caption"
LABEL_FOOTNOTE = "Footnote"
LABEL_TOC = "TableOfContents"

DROP_LABELS = frozenset({LABEL_PAGE_HEADER, LABEL_PAGE_FOOTER})
FIGURE_LABELS = frozenset({LABEL_PICTURE, LABEL_FIGURE})

# Figure / caption markers survive pass-1 and pass-2 (HTML comments are kept).
FIGURE_OPEN = "<!-- figure {fig_id} -->"
FIGURE_CLOSE = "<!-- /figure -->"
CAPTION_OPEN = "<!-- caption -->"
CAPTION_CLOSE = "<!-- /caption -->"

REVIEW_ORPHAN_BEGIN = "<!-- BEGIN USER REVIEW NEEDED: orphan OCR lines -->"
REVIEW_ORPHAN_END = "<!-- END USER REVIEW NEEDED -->"
REVIEW_LOW_CONF_BEGIN = (
    "<!-- BEGIN USER REVIEW NEEDED: low OCR confidence ({conf:.1f}) -->"
)
REVIEW_LOW_CONF_END = "<!-- END USER REVIEW NEEDED -->"

DEFAULT_LOW_CONFIDENCE_THRESHOLD = 70.0  # mean line confidence 0-100 scale
DEFAULT_CENTER_PAD = 6.0
DEFAULT_ROW_Y_TOLERANCE = 12.0


@dataclass
class LayoutBlock:
    label: str
    position: int
    bbox: tuple[float, float, float, float]
    lines: list[SuryaLine] = field(default_factory=list)


@dataclass
class LayoutPageResult:
    """Markdown body plus provenance for lineage."""

    markdown: str
    mean_confidence: float
    layout_class: str
    dropped_headers: list[str] = field(default_factory=list)
    dropped_footers: list[str] = field(default_factory=list)
    orphan_count: int = 0
    low_confidence: bool = False
    block_labels: list[str] = field(default_factory=list)


def line_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def center_in_box(
    line_bbox: tuple[float, float, float, float],
    box_bbox: tuple[float, float, float, float],
    pad: float = DEFAULT_CENTER_PAD,
) -> bool:
    cx, cy = line_center(line_bbox)
    return (
        box_bbox[0] - pad <= cx <= box_bbox[2] + pad
        and box_bbox[1] - pad <= cy <= box_bbox[3] + pad
    )


def assign_lines_to_blocks(
    lines: list[SuryaLine],
    blocks: list[LayoutBlock],
    *,
    pad: float = DEFAULT_CENTER_PAD,
) -> tuple[list[LayoutBlock], list[SuryaLine]]:
    """
    Assign each line to the first layout block (by position order) that
    contains its center. Returns (blocks_with_lines, orphans).
    """
    ordered = sorted(blocks, key=lambda b: b.position)
    used: set[int] = set()
    for block in ordered:
        members: list[SuryaLine] = []
        for i, ln in enumerate(lines):
            if i in used:
                continue
            if center_in_box(ln.bbox, block.bbox, pad=pad):
                members.append(ln)
                used.add(i)
        members.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        block.lines = members
    orphans = [ln for i, ln in enumerate(lines) if i not in used]
    orphans.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return ordered, orphans


def group_table_rows(
    lines: list[SuryaLine],
    *,
    y_tol: float = DEFAULT_ROW_Y_TOLERANCE,
) -> list[list[SuryaLine]]:
    """Group table lines into rows by y-center, cells left-to-right per row."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda ln: (line_center(ln.bbox)[1], ln.bbox[0]))
    rows: list[list[SuryaLine]] = []
    current: list[SuryaLine] = []
    current_y: Optional[float] = None
    for ln in sorted_lines:
        cy = line_center(ln.bbox)[1]
        if current_y is None or abs(cy - current_y) <= y_tol:
            current.append(ln)
            if current_y is None:
                current_y = cy
            else:
                current_y = (current_y * (len(current) - 1) + cy) / len(current)
        else:
            current.sort(key=lambda x: x.bbox[0])
            rows.append(current)
            current = [ln]
            current_y = cy
    if current:
        current.sort(key=lambda x: x.bbox[0])
        rows.append(current)
    return rows


def nearest_figure_index(
    caption: LayoutBlock,
    figures: list[tuple[int, LayoutBlock]],
) -> Optional[int]:
    """Return the list index of the nearest Picture/Figure by bbox center distance."""
    if not figures:
        return None
    cx, cy = line_center(caption.bbox)
    best_i = 0
    best_d = float("inf")
    for i, (_pos, fig) in enumerate(figures):
        fx, fy = line_center(fig.bbox)
        d = (fx - cx) ** 2 + (fy - cy) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _lines_text(lines: list[SuryaLine]) -> str:
    return " ".join(ln.text.strip() for ln in lines if ln.text.strip())


def emit_section_header(lines: list[SuryaLine]) -> str:
    text = _smart_join([ln.text for ln in lines if ln.text.strip()])
    if not text:
        return ""
    return f"### {text}"


def emit_text(lines: list[SuryaLine]) -> str:
    return _join_prose([ln.text for ln in lines if ln.text.strip()])


def emit_list_item(lines: list[SuryaLine]) -> str:
    text = _smart_join([ln.text for ln in lines if ln.text.strip()])
    if not text:
        return ""
    if text.lstrip().startswith(("-", "*", "+")):
        return text
    return f"- {text}"


def emit_table(lines: list[SuryaLine], *, y_tol: float = DEFAULT_ROW_Y_TOLERANCE) -> str:
    rows = group_table_rows(lines, y_tol=y_tol)
    out: list[str] = []
    for row in rows:
        cells = [ln.text.strip() for ln in row if ln.text.strip()]
        if cells:
            out.append(" | ".join(cells))
    return "\n".join(out)


def emit_figure(
    lines: list[SuryaLine],
    *,
    fig_id: str,
) -> str:
    parts = [FIGURE_OPEN.format(fig_id=fig_id)]
    body = _smart_join([ln.text for ln in lines if ln.text.strip()])
    if body:
        parts.append(body)
    parts.append(FIGURE_CLOSE)
    return "\n".join(parts)


def emit_caption(lines: list[SuryaLine]) -> str:
    text = _smart_join([ln.text for ln in lines if ln.text.strip()])
    parts = [CAPTION_OPEN]
    if text:
        parts.append(text)
    parts.append(CAPTION_CLOSE)
    return "\n".join(parts)


def emit_generic(lines: list[SuryaLine]) -> str:
    """Footnote / TOC / unknown: preserve text as a paragraph."""
    return emit_text(lines)


def infer_layout_class(labels: list[str]) -> str:
    non_drop = [lb for lb in labels if lb not in DROP_LABELS]
    if LABEL_TABLE in non_drop:
        return "dense_ingredient_grid"
    if any(lb in FIGURE_LABELS for lb in non_drop) and LABEL_CAPTION in non_drop:
        return "illustration_bleed"
    if LABEL_TOC in non_drop:
        return "toc"
    textish = sum(1 for lb in non_drop if lb in (LABEL_TEXT, LABEL_LIST_ITEM, LABEL_SECTION_HEADER))
    if textish >= 2:
        # Heuristic: multiple text-ish blocks often means multi-column recipe.
        return "two_column_recipe" if textish >= 4 else "single_column_prose"
    if textish:
        return "single_column_prose"
    return "unknown"


def assemble_layout_markdown(
    blocks: list[LayoutBlock],
    orphans: list[SuryaLine],
    *,
    page_num: int = 0,
    mean_confidence: float = 0.0,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    y_tol: float = DEFAULT_ROW_Y_TOLERANCE,
) -> LayoutPageResult:
    """
    Emit blocks in layout position order with caption attachment and
    header/footer drop. Orphans and low confidence get review flags.
    """
    ordered = sorted(blocks, key=lambda b: b.position)

    dropped_headers: list[str] = []
    dropped_footers: list[str] = []
    all_labels = [b.label for b in ordered]

    figures_for_nearest: list[tuple[int, LayoutBlock]] = [
        (b.position, b) for b in ordered if b.label in FIGURE_LABELS
    ]
    # Map figure position -> caption markdown (nearest figure wins).
    caption_by_figure_pos: dict[int, str] = {}
    unattached_captions: list[tuple[int, str]] = []
    for block in ordered:
        if block.label != LABEL_CAPTION:
            continue
        cap_md = emit_caption(block.lines)
        idx = nearest_figure_index(block, figures_for_nearest)
        if idx is None:
            unattached_captions.append((block.position, cap_md))
        else:
            fig_pos = figures_for_nearest[idx][0]
            caption_by_figure_pos[fig_pos] = cap_md

    fig_counter = 0
    chunks: list[str] = []

    for block in ordered:
        if block.label == LABEL_PAGE_HEADER:
            dropped_headers.append(_lines_text(block.lines))
            continue
        if block.label == LABEL_PAGE_FOOTER:
            dropped_footers.append(_lines_text(block.lines))
            continue
        if block.label == LABEL_CAPTION:
            # Attached captions emit with their figure; unattached emit in place.
            idx = nearest_figure_index(block, figures_for_nearest)
            if idx is not None:
                continue
            for pos, cap_md in unattached_captions:
                if pos == block.position and cap_md:
                    chunks.append(cap_md)
            continue
        if block.label in FIGURE_LABELS:
            fig_counter += 1
            fig_id = f"p{page_num:04d}-{fig_counter}" if page_num else f"fig-{fig_counter}"
            chunks.append(emit_figure(block.lines, fig_id=fig_id))
            cap = caption_by_figure_pos.get(block.position)
            if cap:
                chunks.append(cap)
            continue
        if block.label == LABEL_SECTION_HEADER:
            md = emit_section_header(block.lines)
        elif block.label == LABEL_LIST_ITEM:
            md = emit_list_item(block.lines)
        elif block.label == LABEL_TABLE:
            md = emit_table(block.lines, y_tol=y_tol)
        elif block.label == LABEL_TEXT:
            md = emit_text(block.lines)
        else:
            md = emit_generic(block.lines)
        if md:
            chunks.append(md)

    if orphans:
        orphan_body = _join_prose([ln.text for ln in orphans if ln.text.strip()])
        if orphan_body:
            chunks.append(REVIEW_ORPHAN_BEGIN)
            chunks.append(orphan_body)
            chunks.append(REVIEW_ORPHAN_END)

    low_conf = mean_confidence > 0 and mean_confidence < low_confidence_threshold
    if low_conf:
        chunks.insert(0, REVIEW_LOW_CONF_BEGIN.format(conf=mean_confidence))
        chunks.append(REVIEW_LOW_CONF_END)

    markdown = "\n\n".join(c for c in chunks if c).strip()
    return LayoutPageResult(
        markdown=markdown,
        mean_confidence=mean_confidence,
        layout_class=infer_layout_class(all_labels),
        dropped_headers=[h for h in dropped_headers if h],
        dropped_footers=[f for f in dropped_footers if f],
        orphan_count=len(orphans),
        low_confidence=low_conf,
        block_labels=all_labels,
    )


class SuryaLayoutEngine:
    """Lazy-loaded layout + recognition predictors."""

    def __init__(
        self,
        *,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        center_pad: float = DEFAULT_CENTER_PAD,
        row_y_tolerance: float = DEFAULT_ROW_Y_TOLERANCE,
    ) -> None:
        self.low_confidence_threshold = low_confidence_threshold
        self.center_pad = center_pad
        self.row_y_tolerance = row_y_tolerance
        self._layout = None
        self._recognition = None
        self._detection = None

    def _load(self) -> None:
        if self._layout is not None:
            return
        log.info("Loading Surya layout + recognition models...")
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings

        self._layout = LayoutPredictor(
            FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        )
        self._recognition = RecognitionPredictor(FoundationPredictor())
        self._detection = DetectionPredictor()
        log.info("Surya layout models ready.")

    def _bgr_to_pil(self, bgr: np.ndarray) -> Image.Image:
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            rgb = bgr[:, :, ::-1]
        else:
            rgb = bgr
        return Image.fromarray(rgb)

    def ocr_page(
        self,
        bgr: np.ndarray,
        *,
        page_num: int = 0,
    ) -> LayoutPageResult:
        results = self.ocr_pages([bgr], page_nums=[page_num], batch_size=1)
        return results[0]

    def ocr_pages(
        self,
        bgr_images: list[np.ndarray],
        *,
        page_nums: Optional[list[int]] = None,
        batch_size: int = 2,
    ) -> list[LayoutPageResult]:
        if not bgr_images:
            return []
        self._load()
        if batch_size < 1:
            batch_size = 1
        if page_nums is None:
            page_nums = [0] * len(bgr_images)
        out: list[LayoutPageResult] = []
        for start in range(0, len(bgr_images), batch_size):
            chunk = bgr_images[start : start + batch_size]
            nums = page_nums[start : start + batch_size]
            out.extend(self._ocr_page_batch(chunk, nums))
            SuryaEngine.clear_mps_cache()
        return out

    def _ocr_page_batch(
        self,
        bgr_images: list[np.ndarray],
        page_nums: list[int],
    ) -> list[LayoutPageResult]:
        pils = [self._bgr_to_pil(bgr) for bgr in bgr_images]
        layouts = self._layout(pils)
        ocrs = self._recognition(
            pils, det_predictor=self._detection, sort_lines=False
        )
        results: list[LayoutPageResult] = []
        for page_num, lay, ocr in zip(page_nums, layouts, ocrs):
            blocks = [
                LayoutBlock(
                    label=str(b.label),
                    position=int(b.position),
                    bbox=tuple(b.bbox),
                )
                for b in lay.bboxes
            ]
            lines: list[SuryaLine] = []
            confs: list[float] = []
            for ln in ocr.text_lines:
                text = _clean_surya_text(ln.text)
                if not text.strip():
                    continue
                conf = float(ln.confidence or 0)
                lines.append(
                    SuryaLine(text=text, bbox=tuple(ln.bbox), confidence=conf)
                )
                confs.append(conf)
            mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
            assigned, orphans = assign_lines_to_blocks(
                lines, blocks, pad=self.center_pad
            )
            result = assemble_layout_markdown(
                assigned,
                orphans,
                page_num=page_num,
                mean_confidence=mean_conf,
                low_confidence_threshold=self.low_confidence_threshold,
                y_tol=self.row_y_tolerance,
            )
            results.append(result)
        return results
