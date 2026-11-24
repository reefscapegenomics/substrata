"""One page per annotation: an overview photo zoomed on the point, its close-up (macro) photo, and that photo at
100% (``substrata images --layout pages``).

Layout (A4 landscape at 300 dpi by default, ``settings.PAGES_*``)::

    title: plot · annotation · label
    subtitle: annotation id, depth, coordinates, label source, ...
    +---------------------------+  +---------------------------+
    | overview, zoomed on the   |  | close-up, full frame;     |
    | point (scale bar)         |  | box = the crop below      |
    +---------------------------+  +---------------------------+
    caption (photo, pixel, ...)    caption
    +----------------------------------------------------------+
    | close-up at 100% (1 photo pixel = 1 page dot), centred    |
    | on the point (scale bar)                                  |
    +----------------------------------------------------------+

Without a close-up the top right shows the overview's full frame and the bottom the overview at 100%. The point is
marked in every panel. The overview zoom is ``zoom_m`` wide at least, or ``settings.PAGES_ZOOM_CLOSEUP`` times the
close-up's frame width, so the colony the close-up shows is in view. Contrast (:func:`substrata.imageadjust.
auto_contrast`) is taken per photo from what its top panel shows, so a 100% crop matches its full frame.

Pages are rendered as PIL images (:func:`annotation_page`) and written with :func:`save_pages_pdf`, which embeds
each page as one JPEG at the page size (no resampling, so 100% stays 100%) and keeps one page in memory at a time.
"""

# Standard Library
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

# Third-Party Libraries
import numpy as np

# Local Modules
import substrata.settings as settings  # (direct: `from substrata import` would load the whole package)
from substrata.imageadjust import auto_contrast  # noqa: E402
from substrata.compat import renamed_kwargs as _renamed_kwargs  # noqa: E402

_INK, _MUTED, _MARK, _WHITE, _BLACK = (31, 31, 30), (110, 108, 100), (42, 120, 214), (255, 255, 255), (0, 0, 0)
_NICE_M = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)  # scale-bar lengths


@dataclass
class PageView:
    """One photo of a page: the point's pixel in it and its scale there.

    Attributes:
        filepath: The photo.
        x: The point's pixel column.
        y: The point's pixel row.
        m_per_px: Metres per photo pixel at the point (camera distance x world scale / fx); None: no scale bar
            and a zoom of ``settings.PAGES_ZOOM_FRACTION`` of the photo.
        title: First caption line, before the file name (e.g. ``"Overview"``).
        notes: Second caption line.
    """

    filepath: str
    x: float
    y: float
    m_per_px: Optional[float] = None
    title: str = ""
    notes: str = ""


def closeup_camera_types(stored_types: Iterable[Optional[str]], set_type: Optional[str],
                         main_type: Optional[str]) -> set:
    """The camera types whose photos an annotation set's stored views (``cam_filepath``) show as close-ups.

    A set of camera targets (``set_type`` ``camera_targets``, from ``substrata cams2annos``) stores the photo
    aimed at each point: those photos are the close-ups. Otherwise stored views of a camera type other than the
    main one (the type that sees most points best) count as close-ups, e.g. macro photos added to a set by hand;
    stored training views (main type) do not. Returns an empty set when there are none.
    """
    types = {t for t in stored_types if t}
    if set_type == "camera_targets":
        return types
    return types - {main_type}


def nice_scale_length(max_m: float) -> float:
    """The longest round scale-bar length (1-2-5 steps, 1 mm to 5 m) not over ``max_m`` (else 1 mm)."""
    ok = [n for n in _NICE_M if n <= max_m * (1 + 1e-9)]
    return ok[-1] if ok else _NICE_M[0]


def scale_label(length_m: float) -> str:
    """``"5 cm"``, ``"2 mm"``, ``"1 m"``."""
    if length_m >= 1:
        return f"{length_m:g} m"
    return f"{length_m * 100:g} cm" if length_m >= 0.01 else f"{length_m * 1000:g} mm"


def zoom_window(x: float, y: float, width_px: float, aspect: float,
                size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """A ``width_px`` wide window of ``aspect`` (width / height) centred on (x, y), shrunk to fit a photo of
    ``size`` (w, h) and moved inside it: ``(x0, y0, x1, y1)``."""
    w, h = size
    zw = min(float(width_px), w, h * aspect)
    zh = zw / aspect
    zw, zh = max(int(round(zw)), 1), max(int(round(zh)), 1)
    x0 = int(np.clip(round(x - zw / 2), 0, w - zw))
    y0 = int(np.clip(round(y - zh / 2), 0, h - zh))
    return x0, y0, x0 + zw, y0 + zh


def _font(px: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("DejaVuSans-Bold.ttf", "Arial Bold.ttf") if bold else ("DejaVuSans.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default(size=px)


def _mark(draw, x: float, y: float, r: float, width: int) -> None:
    """A ring with four ticks around (x, y), white-edged."""
    draw.ellipse((x - r, y - r, x + r, y + r), outline=_WHITE, width=width + 4)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=_MARK, width=width)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        draw.line((x + dx * r * 1.4, y + dy * r * 1.4, x + dx * r * 2.2, y + dy * r * 2.2), fill=_MARK, width=width)


def _box(draw, xy, width: int) -> None:
    draw.rectangle(xy, outline=_BLACK, width=width + 2)
    draw.rectangle(xy, outline=_WHITE, width=width)


def _scale_bar(img, m_per_px: Optional[float], font, frac: float = 0.15) -> None:
    """A white scale bar about ``frac`` of the width, bottom left; ``m_per_px`` as ``img`` is shown."""
    if not m_per_px or not np.isfinite(m_per_px) or m_per_px <= 0:
        return
    from PIL import ImageDraw

    length = nice_scale_length(frac * img.width * m_per_px)
    d = ImageDraw.Draw(img)
    pad, th = max(img.width // 60, 12), max(img.height // 90, 8)
    x0, y0, x1 = pad, img.height - pad - th, pad + length / m_per_px
    d.rectangle((x0 - 3, y0 - 3, x1 + 3, y0 + th + 3), fill=_BLACK)
    d.rectangle((x0, y0, x1, y0 + th), fill=_WHITE)
    ty = y0 - font.size - th
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        d.text((x0 + ox, ty + oy), scale_label(length), fill=_BLACK, font=font)
    d.text((x0, ty), scale_label(length), fill=_WHITE, font=font)


def _fit(img, box_w: int, box_h: int):
    from PIL import Image

    s = min(box_w / img.width, box_h / img.height)
    return img.resize((max(int(img.width * s), 1), max(int(img.height * s), 1)), Image.LANCZOS), s


def annotation_page(
    title: str,
    subtitle: str,
    overview: Optional[PageView],
    closeup: Optional[PageView] = None,
    zoom_m: float = settings.PAGES_ZOOM_M,
    contrast: Optional[str] = settings.CONTRAST_METHOD_DEFAULT,
    dpi: int = settings.PAGES_DPI,
    size_mm: Tuple[float, float] = settings.PAGES_SIZE_MM,
):
    """One annotation's page (see the module docstring) as a PIL RGB image.

    Args:
        title: Header line (e.g. plot · annotation · label).
        subtitle: Second header line (ids, depth, coordinates, label source, ...).
        overview: The overview view (None: a note instead of the top-left panel).
        closeup: The close-up view (None: the overview fills the page).
        zoom_m: Overview zoom width (m), at least.
        contrast: Contrast method for the photos (:func:`substrata.imageadjust.auto_contrast`).
        dpi: Page raster; the bottom panel is at 100% of its photo at this resolution.
        size_mm: Page width and height (mm).
    """
    from PIL import Image, ImageDraw

    W, H = (int(round(v / 25.4 * dpi)) for v in size_mm)
    pt = dpi / 72.0
    M, G = int(5 * dpi / 25.4), int(5 * dpi / 25.4)
    f_title, f_sub, f_cap, f_bar = (_font(int(11 * pt), True), _font(int(6.2 * pt)), _font(int(7.2 * pt)),
                                    _font(int(7.2 * pt)))
    page = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(page)
    dr.text((M, M - int(2 * pt)), title, fill=_INK, font=f_title)
    dr.text((M, M + int(12.5 * pt)), subtitle, fill=_MUTED, font=f_sub)
    top = M + int(27 * pt)
    bw = (W - 2 * M - G) // 2
    bh = int(bw / 1.5)
    cap_h = int(24 * pt)
    ring, lw = max(int(7 * pt), 6), max(int(1.2 * pt), 2)

    def caption(x, view, extra):
        dr.text((x, top + bh + int(2 * pt)), f"{view.title}  {os.path.basename(view.filepath)}", fill=_INK,
                font=f_cap)
        line = "   ".join(s for s in (extra, view.notes) if s)
        dr.text((x, top + bh + int(11.5 * pt)), line, fill=_MUTED, font=f_sub)

    def open_rgb(path):
        return Image.open(path).convert("RGB")

    # the photo shown at 100% below, and its full frame top right
    main = closeup if closeup is not None else overview
    bottom_w, bottom_h = W - 2 * M, H - (top + bh + cap_h) - M
    ov_img = open_rgb(overview.filepath) if overview is not None else None
    # top left: overview zoom
    if overview is not None:
        if overview.m_per_px:
            want_m = zoom_m
            if closeup is not None and closeup.m_per_px:
                with Image.open(closeup.filepath) as cim:  # (reads the header only)
                    cw = cim.width
                want_m = max(zoom_m, settings.PAGES_ZOOM_CLOSEUP * cw * closeup.m_per_px)
            zoom_px = want_m / overview.m_per_px
        else:
            zoom_px = settings.PAGES_ZOOM_FRACTION * ov_img.width
        x0, y0, x1, y1 = zoom_window(overview.x, overview.y, zoom_px, bw / bh, ov_img.size)
        z, s = _fit(ov_img.crop((x0, y0, x1, y1)), bw, bh)
        z = auto_contrast(z, contrast)
        _mark(ImageDraw.Draw(z), (overview.x - x0) * s, (overview.y - y0) * s, ring, lw)
        if overview.m_per_px:
            _scale_bar(z, overview.m_per_px / s, f_bar)
        page.paste(z, (M, top))
        width_note = (f"zoom {(x1 - x0) * overview.m_per_px:.2f} m wide" if overview.m_per_px
                      else f"zoom {x1 - x0} px wide")
        caption(M, overview, f"{width_note}   point px {int(overview.x)}, {int(overview.y)}")
    else:
        dr.text((M, top + int(10 * pt)), "No overview photo sees this point.", fill=(170, 0, 0), font=f_cap)
    # top right: the main photo's full frame, box = the 100% crop; bottom: that crop
    if main is not None:
        im = ov_img if main is overview else open_rgb(main.filepath)
        ref = im.resize((max(im.width // 8, 1), max(im.height // 8, 1)))
        x0, y0, x1, y1 = zoom_window(main.x, main.y, bottom_w, bottom_w / bottom_h, im.size)
        t, s = _fit(im, bw, bh)
        t = auto_contrast(t, contrast, reference=ref)
        td = ImageDraw.Draw(t)
        _mark(td, main.x * s, main.y * s, ring, lw)
        _box(td, (x0 * s, y0 * s, x1 * s, y1 * s), lw)
        if main.m_per_px:
            _scale_bar(t, main.m_per_px / s, f_bar)
        page.paste(t, (M + bw + G, top))
        caption(M + bw + G, main, f"point px {int(main.x)}, {int(main.y)}   box = 100% crop below")
        cr = auto_contrast(im.crop((x0, y0, x1, y1)), contrast, reference=ref)
        _mark(ImageDraw.Draw(cr), main.x - x0, main.y - y0, ring * 0.8, max(lw - 1, 2))
        if main.m_per_px:
            _scale_bar(cr, main.m_per_px, f_bar, frac=0.08)
        page.paste(cr, (M, H - M - cr.height))
    return page


@_renamed_kwargs(output_filepath="output_path")
def save_pages_pdf(
    pages: Iterable,
    output_path: str,
    size_mm: Tuple[float, float] = settings.PAGES_SIZE_MM,
    quality: int = settings.PAGES_JPEG_QUALITY,
) -> int:
    """Write PIL page images to a PDF, one page each at ``size_mm`` (a generator keeps one page in memory).

    Each page is embedded as a JPEG of its own pixels (not resampled). Returns the number of pages.
    """
    from fpdf import FPDF

    w, h = size_mm
    pdf = FPDF(orientation="L" if w > h else "P", unit="mm", format=(min(w, h), max(w, h)))
    pdf.set_auto_page_break(False)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for page in pages:
            path = os.path.join(tmp, f"page_{n:05d}.jpg")
            page.save(path, quality=quality)
            pdf.add_page()
            pdf.image(path, x=0, y=0, w=w, h=h)
            n += 1
        if n:
            pdf.output(output_path)
    return n
