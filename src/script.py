"""
Cookbook OCR pipeline — accuracy-first.

Scans PDFs in ../input, rasterizes with PyMuPDF, preprocesses with OpenCV,
runs Tesseract via pytesseract with OSD rotation + dual-PSM confidence
picking, writes one Markdown file per PDF to ../output.

Requires:
    pip install pymupdf pillow pytesseract opencv-python numpy
    Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
    For best accuracy on old print, also install the `eng_best` traineddata:
        https://github.com/tesseract-ocr/tessdata_best
        drop eng.traineddata into a folder and point --tessdata at it,
        then run with --lang eng.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from wordfreq import zipf_frequency

# Windows: auto-detect standard Tesseract install.
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(DEFAULT_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
# Auto-detect tessdata_best at project root. Much higher accuracy than the
# default "fast" model shipped with UB-Mannheim, especially for fractions
# and small serif type common in cookbooks.
LOCAL_TESSDATA_BEST = PROJECT_DIR / "tessdata_best"

RENDER_DPI = 400                      # high DPI for small serif in old cookbooks
MIN_EFFECTIVE_DPI = 320               # upscale below this
TESS_LANG = "eng"
TESS_OEM = 1                          # LSTM only
# Page segmentation modes to race. 1 = auto with OSD, 3 = auto, 4 = single
# column variable sizes, 6 = uniform block of text (wins for dense body copy).
TESS_PSM_CANDIDATES = (1, 3, 4, 6)

LIGATURE_MAP = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
    "\u2014": "--", "\u2013": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ",
}

# Common Tesseract misreads for vulgar fractions. Fired only when the token
# is adjacent to a measurement context (digit before, or unit word after) to
# avoid corrupting real text. Patterns tuned against observed output.
_UNIT = r"(?:cup|tsp|tbsp|teaspoon|tablespoon|pound|oz|lb|quart|pint|gallon|cloves?|stick)"

FRACTION_FIXUPS: list[tuple[re.Pattern, str]] = [
    # Bare "¥%", "¥/" — yen/percent combo that Tesseract emits for ½.
    (re.compile(r"¥\s*[%/]"), "½"),
    # "Ys" followed by a unit → ¼ (stylized ¼ glyph rendered as Ys).
    (re.compile(rf"\bYs\b(?=\s+{_UNIT})", re.I), "¼"),
    # "'/«", "'/<", "'/.", "'/," → ¼ (fancy ¼ glyph).
    (re.compile(r"['`]\s*[/\\]\s*[«<.,]"), "¼"),
    # Leading "%." at start of a measurement → ¼.
    (re.compile(rf"(?m)^\s*%\.(?=\s+{_UNIT})", re.I), "¼"),
    # Leading "%" at start of a measurement → ½.
    (re.compile(rf"(?m)^\s*%(?=\s+{_UNIT})", re.I), "½"),
    # "\d½\d" (stuck digit after recognized ½) → drop trailing digit.
    (re.compile(r"(\d½)\d\b"), r"\1"),
    # "1'%", "1'/", etc. → "1½".
    (re.compile(r"(\d)['`]?\s*[%/][.,]?(?=\s)"), r"\1½"),
    # "4%" before a unit → ¼.
    (re.compile(rf"(?<!\d)4%(?=\s+{_UNIT})", re.I), "¼"),
    # "%" between digit and unit ("2% cup") → ½.
    (re.compile(rf"(?<=\s)%(?=\s+{_UNIT})", re.I), "½"),
    # ASCII fractions adjacent to measurement units → Unicode vulgar fractions.
    (re.compile(rf"\b1/2\b(?=\s+{_UNIT})", re.I), "½"),
    (re.compile(rf"\b1/4\b(?=\s+{_UNIT})", re.I), "¼"),
    (re.compile(rf"\b3/4\b(?=\s+{_UNIT})", re.I), "¾"),
    (re.compile(rf"\b1/3\b(?=\s+{_UNIT})", re.I), "⅓"),
    (re.compile(rf"\b2/3\b(?=\s+{_UNIT})", re.I), "⅔"),
    (re.compile(rf"\b1/8\b(?=\s+{_UNIT})", re.I), "⅛"),
    # Composite "\d 1/2" → "\d½" (e.g. "1 1/2 cups").
    (re.compile(rf"(\d)\s+1/2(?=\s+{_UNIT})", re.I), r"\1½"),
    (re.compile(rf"(\d)\s+1/4(?=\s+{_UNIT})", re.I), r"\1¼"),
    (re.compile(rf"(\d)\s+3/4(?=\s+{_UNIT})", re.I), r"\1¾"),
    # Mangled ½ variants with leading quote: "'/2 cup", "'/4 cup".
    (re.compile(rf"'\s*/\s*2(?=\s+{_UNIT})", re.I), "½"),
    (re.compile(rf"'\s*/\s*4(?=\s+{_UNIT})", re.I), "¼"),
    # "Y=}" / "Y=}" with any trailing junk before unit → ¼ or ¼-½ range.
    (re.compile(rf"Y\s*=\s*\}}(?=\s*[-–]\s*[½1/]?\s*{_UNIT})", re.I), "¼"),
    (re.compile(rf"(?<!\w)Y[=\.]\s*(?=\s+{_UNIT})", re.I), "¼"),
    # "Y." / "Y," / "Y«" / "Y»" standalone fraction-like → ¼.
    (re.compile(rf"(?<!\w)Y[.,«»<>]+(?=\s+{_UNIT})", re.I), "¼"),
    # "1%:" / "1%," / "1%;" → 1½.
    (re.compile(rf"(\d)%[:;,.]+(?=\s+{_UNIT})", re.I), r"\1½"),
    # "2»" / "2«" at line start before a unit → ½ (mis-OCR of ½ glyph).
    (re.compile(rf"(?m)^\s*2[»«](?=\s+{_UNIT})", re.I), "½"),
    (re.compile(rf"(?m)^\s*1[»«](?=\s+{_UNIT})", re.I), "¼"),
]

log = logging.getLogger("cookbook-ocr")


# ---------- rasterization ----------

def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------- preprocessing ----------

def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate and expand canvas so nothing is cropped."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw / 2) - cx
    m[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def correct_orientation(bgr: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to auto-rotate 90/180/270 scans. No-op on failure."""
    try:
        osd = pytesseract.image_to_osd(bgr, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0)) % 360
        if rot:
            log.debug("OSD rotate %d", rot)
            return rotate_bound(bgr, rot)
    except pytesseract.TesseractError as e:
        log.debug("OSD failed: %s", e)
    return bgr


def remove_rulers(bgr: np.ndarray) -> np.ndarray:
    """
    Inpaint long horizontal/vertical ruler lines that confuse Tesseract's
    layout analysis on pages with decorative column separators or margin rules.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = gray.shape
    # Require rulers to be quite long (≥ 12% of dimension) so we don't nuke glyphs.
    h_len = max(40, w // 8)
    v_len = max(40, h // 8)
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    )
    rulers = cv2.bitwise_or(horiz, vert)
    if rulers.sum() == 0:
        return bgr
    rulers = cv2.dilate(rulers, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, rulers, 3, cv2.INPAINT_TELEA)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Small-angle skew correction via minAreaRect of text pixels."""
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.2 or abs(angle) > 10:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def ensure_min_dpi(gray: np.ndarray, src_dpi: int, min_dpi: int) -> np.ndarray:
    if src_dpi >= min_dpi:
        return gray
    scale = min_dpi / src_dpi
    h, w = gray.shape
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess(bgr: np.ndarray, src_dpi: int, aggressive: bool = False) -> Image.Image:
    """
    Pipeline tuned for old, yellowed cookbook pages:
      grayscale → CLAHE (uneven lighting) → denoise → deskew → upscale → binarize.

    ``aggressive``: use a tighter hard threshold that only keeps near-black ink.
    Useful when illustrations or screened backgrounds bleed through text boxes.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    gray = deskew(gray)
    gray = ensure_min_dpi(gray, src_dpi, MIN_EFFECTIVE_DPI)
    if aggressive:
        # Hard threshold: only the darkest ink pixels survive. Washes out
        # medium-gray illustration line-art while keeping pure-black text.
        _, binarized = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    else:
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
    return Image.fromarray(binarized)


def detect_column_ranges(bgr: np.ndarray) -> list[tuple[int, int]]:
    """
    Detect text columns from vertical projection of the page.
    Returns a list of (x_start, x_end). Single entry = no split needed.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape
    # Binary inverse: ink = 255, paper = 0.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Ignore horizontal decorative rules: remove very long horizontal ink runs
    # so they don't dominate the projection.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 30), 1))
    horiz_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
    bw = cv2.subtract(bw, horiz_lines)

    col_ink = bw.sum(axis=0) / 255.0
    k = max(5, w // 200)
    smoothed = np.convolve(col_ink, np.ones(k) / k, mode="same")
    empty = smoothed < (h * 0.01)
    # Columns where most of the height is inked are vertical decorative rules —
    # treat them as gutters too (they separate page regions just like whitespace).
    rule = smoothed > (h * 0.55)
    boundary = empty | rule

    nonempty = np.where(~boundary)[0]
    if nonempty.size == 0:
        return [(0, w)]
    left_edge, right_edge = int(nonempty[0]), int(nonempty[-1]) + 1

    # Find internal gutters (wide runs of boundary columns, not touching edges).
    # Rules can be quite thin (3-5px), so allow shorter runs.
    min_gutter = max(8, int(w * 0.005))
    gutters: list[tuple[int, int]] = []
    i = left_edge
    while i < right_edge:
        if boundary[i]:
            j = i
            while j < right_edge and boundary[j]:
                j += 1
            if j - i >= min_gutter:
                gutters.append((i, j))
            i = j
        else:
            i += 1

    if not gutters:
        return [(left_edge, right_edge)]

    cuts = [left_edge] + [(gs + ge) // 2 for gs, ge in gutters] + [right_edge]
    min_col = max(120, int(w * 0.15))
    cols = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
            if cuts[i + 1] - cuts[i] >= min_col]
    return cols or [(left_edge, right_edge)]


# ---------- OCR ----------

def ocr_with_confidence(img: Image.Image, psm: int, lang: str, extra_cfg: str) -> tuple[str, float]:
    """Run Tesseract at given PSM, return (text, mean word confidence)."""
    config = f"--oem {TESS_OEM} --psm {psm} {extra_cfg}".strip()
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    confidences = [int(c) for c in data["conf"] if c not in ("-1", "") and int(c) >= 0]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    # Rebuild text in reading order from the data dict (preserves blocks/paragraphs).
    lines: dict[tuple, list[tuple[int, str]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((data["left"][i], word))
    ordered = []
    prev_block = None
    for key in sorted(lines):
        block = key[0]
        if prev_block is not None and block != prev_block:
            ordered.append("")  # paragraph break between blocks
        ordered.append(" ".join(w for _, w in sorted(lines[key])))
        prev_block = block
    return "\n".join(ordered), mean_conf


def _race_psms(img: Image.Image, lang: str, extra_cfg: str, psms: tuple[int, ...]) -> tuple[str, float, int]:
    best: tuple[str, float, int] = ("", -1.0, -1)
    for psm in psms:
        try:
            text, conf = ocr_with_confidence(img, psm, lang, extra_cfg)
        except pytesseract.TesseractError as e:
            log.debug("PSM %d failed: %s", psm, e)
            continue
        score = conf + (len(text) / 1e6)
        if score > best[1] + (len(best[0]) / 1e6):
            best = (text, conf, psm)
    return best


def best_ocr(img: Image.Image, lang: str, extra_cfg: str) -> tuple[str, float, int]:
    return _race_psms(img, lang, extra_cfg, TESS_PSM_CANDIDATES)


def best_ocr_layout_aware(
    bgr: np.ndarray, src_dpi: int, lang: str, extra_cfg: str
) -> tuple[str, float, str]:
    """
    Detect columns and pick the highest-confidence variant among:
      - full-page (standard adaptive threshold)
      - full-page (aggressive threshold — kills illustration bleed-through)
      - per-column (when 2+ columns detected)
    Returns (text, conf, mode_label).
    """
    variants: list[tuple[str, float, str]] = []

    full_img = preprocess(bgr, src_dpi)
    full_text, full_conf, full_psm = best_ocr(full_img, lang, extra_cfg)
    variants.append((full_text, full_conf, f"full/psm{full_psm}"))

    aggr_img = preprocess(bgr, src_dpi, aggressive=True)
    aggr_text, aggr_conf, aggr_psm = best_ocr(aggr_img, lang, extra_cfg)
    variants.append((aggr_text, aggr_conf, f"hard/psm{aggr_psm}"))

    columns = detect_column_ranges(bgr)
    log.debug("Detected %d column(s): %s", len(columns), columns)
    if len(columns) >= 2:
        # Race both preprocessing modes per column. PSM 1 (auto+OSD) often
        # picks up leading fraction glyphs that PSM 6 drops.
        col_psms = (1, 6, 4)
        col_texts: list[str] = []
        col_confs: list[float] = []
        # Pad each column generously so a fraction glyph or punctuation hugging
        # the gutter edge isn't clipped off the leading or trailing character.
        pad = max(40, int(bgr.shape[1] * 0.02))
        max_x = bgr.shape[1]
        for x0, x1 in columns:
            x0p = max(0, x0 - pad)
            x1p = min(max_x, x1 + pad)
            col_strip = bgr[:, x0p:x1p]
            col_img = preprocess(col_strip, src_dpi)
            col_aggr = preprocess(col_strip, src_dpi, aggressive=True)
            t1, c1, _ = _race_psms(col_img, lang, extra_cfg, col_psms)
            t2, c2, _ = _race_psms(col_aggr, lang, extra_cfg, col_psms)
            if c2 > c1:
                text, conf = t2, c2
            else:
                text, conf = t1, c1
            col_texts.append(text)
            col_confs.append(conf if conf > 0 else 0.0)
        multi_conf = float(np.mean(col_confs)) if col_confs else 0.0
        multi_text = "\n\n".join(t for t in col_texts if t.strip())
        # When multiple columns are detected, reading order matters more than
        # marginal confidence — prefer columns unless they fail badly.
        if multi_text.strip() and multi_conf >= 55:
            return multi_text, multi_conf, f"cols-{len(columns)}"

    # Single-column page (or column mode failed): pick best of full/hard.
    variants.sort(key=lambda v: (v[1], len(v[0])), reverse=True)
    return variants[0]


# ---------- cleanup ----------

_RECIPE_HEADER_WORDS = {
    "ingredients", "directions", "instructions", "method", "notes", "serves",
    "yield", "prep", "cook", "total", "serving", "servings", "makes",
}

# Cookbook terms that are real English but have low corpus frequency.
_COOKBOOK_ALLOW = {
    "tahini", "gochujang", "harissa", "miso", "dashi", "mirin", "shoyu",
    "za'atar", "sumac", "urfa", "aleppo", "chipotle", "ancho", "guajillo",
    "saffron", "cardamom", "fennel", "coriander", "sumac", "pomegranate",
    "scampi", "quinoa", "couscous", "bulgur", "freekeh", "farro",
}


def _looks_like_word(token: str) -> bool:
    """Plausible-English-word check using wordfreq + structural rules."""
    t = token.lower().strip(".,!?;:'\"()[]{}/\\")
    if len(t) < 2:
        return False
    if any(ch.isdigit() or ch.isnumeric() for ch in t):
        return True  # numbers, fractions, and measurements always count
    if t in _COOKBOOK_ALLOW or t in _RECIPE_HEADER_WORDS:
        return True
    if not any(c in "aeiouy" for c in t):
        return False
    if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", t):
        return False
    # wordfreq zipf: 7=ultra-common, 3=moderate, 2=rare but real, 0=unknown.
    return zipf_frequency(t, "en") >= 2.0


_SINGLE_CHAR_WORDS = {"a", "i", "o", "&", "+"}


def _has_numeric(token: str) -> bool:
    """Treats digits and vulgar fractions (½, ¼, etc.) as numeric content."""
    return any(ch.isdigit() or ch.isnumeric() for ch in token)


def _strip_line_suffix_junk(line: str) -> str:
    """Drop trailing margin-rule artifacts like 'day or |' or '--H' at line end."""
    s = line.rstrip()
    s = re.sub(r"\s+[|:]+\s*$", "", s)
    tokens = s.split()
    drop = 0
    for t in reversed(tokens[-3:]):
        bare = t.strip(".,!?;:|/\\<>*'\"()[]{}-_")
        if not bare:
            drop += 1
            continue
        if bare.lower() in _SINGLE_CHAR_WORDS:
            break
        if _looks_like_word(bare) or _has_numeric(bare):
            break
        if len(bare) <= 2:
            drop += 1
            continue
        break
    if drop:
        return " ".join(tokens[: len(tokens) - drop])
    return s


def _strip_line_prefix_junk(line: str) -> str:
    """Drop leading single-char or pipe-like tokens that came from margin rules."""
    s = line.lstrip()
    s = re.sub(r"^[|:]+\s+", "", s)
    tokens = s.split()
    drop = 0
    for t in tokens[:3]:
        bare = t.strip(".,!?;:|/\\<>*'\"()[]{}-")
        if not bare:
            drop += 1
            continue
        if bare.lower() in _SINGLE_CHAR_WORDS:
            break
        if _looks_like_word(bare) or _has_numeric(bare):
            break
        if len(bare) <= 2:  # "Co", "Ho" and similar short non-words
            drop += 1
            continue
        break
    if drop:
        return " ".join(tokens[drop:])
    return s


def _is_garbage_line(line: str) -> bool:
    """Detect decorative borders / OCR noise from ornaments."""
    stripped = line.strip().rstrip(" ")
    if not stripped:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    total = len(stripped)
    if total <= 2 and letters <= 1:
        return True
    if total >= 6 and letters / total < 0.35:
        return True

    tokens = stripped.split()
    word_tokens = [t for t in tokens if _looks_like_word(t)]
    bare_tokens = [t.strip(".,!?;:'\"()[]{}") for t in tokens]

    # Very short lines with no digit = isolated ornament debris ("Ho HE", "a uA").
    # Digits protect legit ingredient starts ("1 tsp", "4 oz").
    if total < 7 and not any(ch.isdigit() for ch in stripped):
        return True
    # Short lines where no token is a plausible word.
    if len(tokens) <= 4 and total < 30 and not word_tokens:
        return True
    # Isolated single-word lines that are all-caps and not known headers.
    if len(tokens) == 1:
        bare = bare_tokens[0]
        if bare.isupper() and len(bare) <= 8 and bare.lower() not in _RECIPE_HEADER_WORDS:
            if zipf_frequency(bare.lower(), "en") < 3.5:
                return True
    # Short all-caps lines where no token is a common English word → ornament.
    # "BEANE NRA RENE NERA", "EEE HEHE", etc.
    if total < 35 and all(t.isupper() and len(t) >= 2 for t in bare_tokens if t):
        common = [t for t in bare_tokens if zipf_frequency(t.lower(), "en") >= 4.0]
        if len(common) < max(1, len(bare_tokens) // 2):
            return True
    # Runs of short tokens with <40% recognizable.
    if len(tokens) >= 2 and len(word_tokens) / max(1, len(tokens)) < 0.4:
        return True
    return False


def clean_text(raw: str) -> str:
    # NFC (not NFKC): NFKC's compatibility decomposition breaks vulgar
    # fraction glyphs like "½" into "1⁄2" (fraction slash), undoing the
    # fraction normalization that the Surya engine does upstream.
    text = unicodedata.normalize("NFC", raw)
    for src, dst in LIGATURE_MAP.items():
        text = text.replace(src, dst)

    lines = [ln.rstrip() for ln in text.splitlines()]
    merged: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Join hyphenated line wraps: "exam-\nple" -> "example", but only when lowercase continues.
        if (
            ln.endswith("-")
            and i + 1 < len(lines)
            and lines[i + 1]
            and lines[i + 1][:1].islower()
        ):
            merged.append(ln[:-1] + lines[i + 1].lstrip())
            i += 2
        else:
            merged.append(ln)
            i += 1

    # Strip leading AND trailing margin debris from each line
    # ("| Add to the pan:" → "Add to the pan:", "day or |" → "day or").
    cleaned_lines: list[str] = []
    for ln in merged:
        stripped = _strip_line_prefix_junk(ln)
        stripped = _strip_line_suffix_junk(stripped)
        cleaned_lines.append(stripped)

    filtered = [ln for ln in cleaned_lines if not _is_garbage_line(ln)]
    text = "\n".join(filtered)

    for pattern, repl in FRACTION_FIXUPS:
        text = pattern.sub(repl, text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    raw_lines = text.splitlines()
    out_lines: list[str] = []
    for i, ln in enumerate(raw_lines):
        stripped = ln.rstrip()
        if not stripped:
            out_lines.append("")
            continue
        prev = raw_lines[i - 1].strip() if i > 0 else ""
        nxt = raw_lines[i + 1].strip() if i + 1 < len(raw_lines) else ""
        # Skip heading candidate if everything after it on the page is blank —
        # that's a page-footer (book title, page number) rather than a title.
        rest_blank = not any(raw_lines[j].strip() for j in range(i + 1, len(raw_lines)))
        if _is_likely_heading(stripped, prev, nxt) and not rest_blank:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(f"## {stripped}")
        else:
            out_lines.append(stripped + "  ")
    return "\n".join(out_lines).strip()


_DATA_LABEL = re.compile(
    r"^\s*\d+\s*(?:[°%]|mL|ml|cm|mm|km|cup|tsp|tbsp|oz|lb|g\b|kg)",
    re.I,
)


def _looks_like_prose(line: str) -> bool:
    """Does the line read as a continuation of natural text (not data/comic)?"""
    if not line:
        return False
    if _DATA_LABEL.match(line):
        return False
    if "|" in line:
        return False
    # Typical prose: starts with a letter, contains lowercase, at least some length.
    if not line[:1].isalpha():
        return False
    lowercase_count = sum(1 for ch in line if ch.islower())
    return lowercase_count >= 3


def _is_likely_heading(line: str, prev: str, nxt: str) -> bool:
    """Detect recipe titles / section headers isolated between blank lines."""
    if len(line) > 80 or len(line) < 6:
        return False
    if line[-1] in ",;.!?":
        return False
    # Must have a blank line above. nxt may be blank OR prose-like — but if
    # it's a data row or comic fragment, this line is probably a fragment too.
    if prev:
        return False
    if nxt and not _looks_like_prose(nxt):
        return False
    if _DATA_LABEL.match(line):
        return False
    # Lines starting with a digit are almost always list items or data rows,
    # not recipe titles. Pipes = comic panels / table separators.
    if line[:1].isdigit() or "|" in line:
        return False
    tokens = line.split()
    if len(tokens) < 2:
        return False
    # Require a lowercase letter somewhere — Title Case, not all caps.
    # All-caps standalone lines in cookbooks are almost always decorative or
    # part of data layouts, never prose-style chapter headings.
    if not any(ch.islower() for ch in line):
        return False
    content = [t for t in tokens if len(t.strip(".,!?;:'\"()[]{}")) > 3]
    if not content:
        return False
    if not all(t[:1].isupper() or not t[:1].isalpha() for t in content):
        return False
    # Need at least two recognizable English words — rejects Tesseract salad
    # that happens to pass Title Case shape.
    real = [t for t in tokens if _looks_like_word(t)]
    if len(real) < 2:
        return False
    return True


def _maybe_clear_mps_after_page(engine_name: str) -> None:
    """Release pooled MPS memory between pages — mitigates long-run growth on Apple Silicon."""
    if engine_name != "surya":
        return
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------- driver ----------

def ocr_pdf(
    pdf_path: Path,
    out_path: Path,
    dpi: int,
    lang: str,
    extra_cfg: str,
    engine_name: str = "tesseract",
    surya_engine=None,
) -> None:
    log.info("Processing %s", pdf_path.name)
    doc = fitz.open(pdf_path)
    total = doc.page_count

    header = [
        f"# {pdf_path.stem}",
        "",
        f"*OCR transcription of `{pdf_path.name}` ({total} pages). Engine: {engine_name}.*",
        "",
        "---",
        "",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header))
        for i, page in enumerate(doc, start=1):
            bgr = render_page(page, dpi)
            bgr = correct_orientation(bgr)
            if engine_name == "surya" and surya_engine is not None:
                from surya_engine import assemble_markdown
                lines, conf = surya_engine.ocr_page(bgr)
                # assemble_markdown emits finished markdown (headings, bullets,
                # joined prose); clean_text would damage the bullet markers.
                cleaned = assemble_markdown(lines, page_width=bgr.shape[1])
                mode = f"surya/{len(lines)}lines"
            else:
                text, conf, mode = best_ocr_layout_aware(bgr, dpi, lang, extra_cfg)
                cleaned = clean_text(text)
            log.info("  page %3d/%d  conf=%.1f  mode=%s  chars=%d", i, total, conf, mode, len(cleaned))
            f.write(f"\n## Page {i}\n\n")
            if cleaned:
                f.write(cleaned)
            else:
                f.write("_(no text detected)_")
            f.write("\n")
            _maybe_clear_mps_after_page(engine_name)
    doc.close()
    log.info("Wrote %s", out_path)


def check_tesseract() -> None:
    cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(cmd) is None and not Path(cmd).exists():
        raise SystemExit(
            "Tesseract not found. Install UB-Mannheim build:\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki\n"
            f"Or set the executable path: pytesseract.pytesseract.tesseract_cmd = ...\n"
            f"Currently: {cmd}"
        )
    log.info("Tesseract %s", pytesseract.get_tesseract_version())


def main() -> int:
    parser = argparse.ArgumentParser(description="Cookbook OCR pipeline.")
    parser.add_argument("--src", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    parser.add_argument("--lang", default=TESS_LANG, help="Tesseract language (e.g. 'eng', 'eng+nld').")
    parser.add_argument("--tessdata", type=Path, default=None, help="Override TESSDATA_PREFIX (e.g. tessdata_best).")
    parser.add_argument("--tesseract", type=Path, default=None, help="Path to tesseract.exe.")
    parser.add_argument("--force", action="store_true", help="Re-OCR even if markdown exists.")
    parser.add_argument(
        "--engine",
        choices=("tesseract", "surya"),
        default="surya",
        help="OCR engine. Surya is transformer-based, higher accuracy, slower.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = str(args.tesseract)

    extra_cfg = ""
    tessdata = args.tessdata
    if tessdata is None and LOCAL_TESSDATA_BEST.exists():
        tessdata = LOCAL_TESSDATA_BEST
        log.info("Using local tessdata_best at %s", tessdata)
    if tessdata:
        if not tessdata.exists():
            raise SystemExit(f"tessdata path not found: {tessdata}")
        tessdata_abs = tessdata.resolve()
        if " " in str(tessdata_abs):
            raise SystemExit(
                "tessdata path contains spaces; move it to a space-free location "
                "(pytesseract config parsing cannot handle quoted paths)."
            )
        os.environ["TESSDATA_PREFIX"] = str(tessdata_abs)
        extra_cfg = f"--tessdata-dir {tessdata_abs}"
        # Cooking-domain wordlist nudges Tesseract toward cookbook vocab
        # (accented French terms, pasta shapes, chile varieties, etc.).
        user_words = tessdata_abs / "cookbook-words.txt"
        if user_words.exists() and " " not in str(user_words):
            extra_cfg += f" --user-words {user_words}"

    surya_engine = None
    if args.engine == "surya":
        from surya_engine import SuryaEngine
        surya_engine = SuryaEngine()
    else:
        check_tesseract()

    args.out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.src.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found in %s", args.src)
        return 1

    for pdf in pdfs:
        md = args.out / f"{pdf.stem}.md"
        if md.exists() and not args.force:
            log.info("Skipping %s (exists; --force to re-run)", md.name)
            continue
        ocr_pdf(pdf, md, args.dpi, args.lang, extra_cfg, args.engine, surya_engine)

    return 0


if __name__ == "__main__":
    sys.exit(main())
