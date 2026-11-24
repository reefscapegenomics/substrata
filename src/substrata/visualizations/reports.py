"""Review PDFs and figures of annotations and scale bars.

Cropped image matches and the random-point -> manual-annotation match review
(``save_cropped_image_matches_to_pdf``, ``save_annotation_matches_to_pdf``),
box plots and per-annotation pages of the stored measurements
(``plot_measurements``, ``save_measurement_visualizations_to_pdf``,
``show_measurement_image(s)``), scale-bar target figures and PDF
(``scalebar_figures``, ``show_scalebars``, ``save_scalebars_pdf``) and
``plot_text`` (a text page). The measurements are those stored on
:class:`substrata.annotations.Annotation` by ``Annotations.measure_all``.
"""

# Standard Library
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from fpdf import FPDF
from PIL import Image
from tqdm import tqdm

# Local Modules
from substrata import settings
from substrata.fileio import imread as _imread
from substrata.imagecrops import encode_to_png_buffer, get_crop_img
from substrata.logging import logger
from substrata.visualizations._common import _draw_placeholder
from substrata.visualizations.clouds import plot_2d
from substrata.visualizations.photos import get_crop_img_from_masks


def plot_text(text, width=10, height=5):
    """
    Plots a text string as a figure (e.g. for inclusion in a PDF).
    """
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.5, text, ha="left", va="center")
    ax.set_axis_off()  # Hide all axes
    ax.set_rasterized(True)
    return fig


def save_cropped_image_matches_to_pdf(
    image_matches,
    output_filepath,
    n_cols=3,
    n_rows=4,
    crop_w=1000,
    crop_h=1000,
    single_mask=False,
    show_text=True,
    show_radius=False,
    show_point=True,
):
    """
    Save a PDF file from a list of ImageMatches

    ``show_text`` / ``show_radius`` / ``show_point`` are passed to
    :func:`get_crop_img_from_masks` for matches with SAM2 masks.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.set_font("Arial", size=8)

    pdf.add_page()
    # Page width and height in FPDF's units (default: mm)
    page_w = pdf.w
    page_h = pdf.h
    margin = 10

    # Compute the usable space (accounting# for margins)
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    # Cell size in the grid
    cell_w = usable_w / n_cols
    cell_h = usable_h / n_rows
    image_h = cell_h * 0.8

    # Track current row and column in the grid
    row_idx = 0
    idx = 0

    for match in image_matches.values():
        # Calculate the (x, y) position for this cell
        x = margin + idx * cell_w
        y = margin + row_idx * cell_h

        if match.masks:
            cropped_img = get_crop_img_from_masks(
                match,
                crop_w,
                crop_h,
                single_mask=single_mask,
                show_text=show_text,
                show_radius=show_radius,
                show_point=show_point,
            )
        else:
            cropped_img = get_crop_img(
                match.cam.filepath, match.x, match.y, crop_w, crop_h
            )
        pdf.image(encode_to_png_buffer(cropped_img), x=x, y=y, w=cell_w, h=image_h)

        # Position the text a bit below the image
        label_x = x + 2  # small offset from left edge
        label_y = y + image_h + 4  # 4 units below bottom of the image
        pdf.text(label_x, label_y, f"{match.annotation.id} {match.annotation.label}")

        # Move to the next column
        idx += 1
        if idx == n_cols:
            idx = 0
            row_idx += 1
            if row_idx == n_rows:
                pdf.add_page()
                row_idx = 0
                idx = 0

    pdf.output(output_filepath)
    print(f"PDF created: {output_filepath}")


def _pil_to_pdf(pdf, pil_img, x, y, max_w, max_h):
    """Place *pil_img* centred inside the cell (x, y, max_w, max_h), preserving aspect ratio."""
    try:
        iw, ih = pil_img.size
        aspect = iw / ih if ih > 0 else 1.0
        cell_aspect = max_w / max_h if max_h > 0 else 1.0
        if aspect > cell_aspect:
            scaled_w = max_w
            scaled_h = max_w / aspect
        else:
            scaled_h = max_h
            scaled_w = max_h * aspect
        img_x = x + (max_w - scaled_w) / 2.0
        img_y = y + (max_h - scaled_h) / 2.0
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, x=img_x, y=img_y, w=scaled_w, h=scaled_h)
    except Exception as e:
        logger.warning("_pil_to_pdf failed: %s", e)
        pdf.rect(x, y, max_w, max_h)


# Review-PDF colours (BGR): the random point is red; candidates get a fixed
# colour per candidate number. 2nd-ranked SAM2 masks use a lighter shade.
_MATCH_RANDOM_COLOR = (0, 0, 230)
_MATCH_CANDIDATE_COLORS = [
    (255, 110, 0),  # 1 blue
    (40, 190, 0),  # 2 green
    (0, 150, 255),  # 3 orange
    (220, 0, 220),  # 4 magenta
    (200, 200, 0),  # 5 cyan
    (0, 215, 255),  # 6 yellow
    (140, 70, 150),  # 7 purple
    (90, 90, 90),  # 8 grey
]


def _match_color(k):
    """BGR colour of candidate ``k`` (1-based); 0 = the random point."""
    if k == 0:
        return _MATCH_RANDOM_COLOR
    return _MATCH_CANDIDATE_COLORS[(k - 1) % len(_MATCH_CANDIDATE_COLORS)]


def _lighter(color, amount=0.55):
    return tuple(int(c + (255 - c) * amount) for c in color)


def _short_ids(ids):
    """Id suffix after the last underscore, unless that makes ids ambiguous."""
    short = [str(i).rsplit("_", 1)[-1] for i in ids]
    return short if len(set(short)) == len(short) else [str(i) for i in ids]


def _vote_str(result):
    if result is None:
        return "-"
    if not result:
        return "no mask"
    if result[0]:
        return "yes"
    if len(result) > 1 and result[1]:
        return "yes (2nd)"
    return "no"


def _render_match_panel(cam, record, labels, masks=(), focus_ks=None, crop_px=600):
    """Crop one camera image for the annotation-match review PDF.

    Draws SAM2 mask contours, the random point (red cross) and every candidate
    manual annotation (numbered colour markers) projected into ``cam``.

    Args:
        cam: Camera whose image is rendered.
        record: Match record from ``match_annotations``.
        labels: Marker label per candidate (same order as the candidates).
        masks: ``(contours, color, thickness)`` tuples to draw.
        focus_ks: Point indices (0 = random point, k = candidate k) the crop
            should contain; None = all visible points.
        crop_px: Output size (px) of the longest side.

    Returns:
        BGR image, or None if the image cannot be read.
    """
    img = _imread(cam.filepath)
    if img is None:
        logger.warning(f"Could not read image: {cam.filepath}")
        return None
    h_img, w_img = img.shape[:2]
    pts = [record["orig_coords"]] + [c["orig_coords"] for c in record["candidates"]]
    x, y, depth, _, in_view = cam.project_points(
        np.array(pts, dtype=float), use_orig_coords=True
    )
    visible = np.asarray(in_view) & (np.asarray(depth) > 0)
    if focus_ks is None:
        focus_ks = range(len(pts))
    focus = [(x[k], y[k]) for k in focus_ks if visible[k]]
    for contours, _, _ in masks:
        for c in contours:
            focus.extend([tuple(c.min(axis=0)), tuple(c.max(axis=0))])
    if not focus:
        focus = [(w_img / 2, h_img / 2)]
    focus = np.array(focus, dtype=float)
    lo, hi = focus.min(axis=0), focus.max(axis=0)
    side = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.3
    side = int(min(max(side, settings.MATCH_REVIEW_MIN_CROP_PX), max(w_img, h_img)))
    cx, cy = (lo + hi) / 2
    left = int(np.clip(cx - side / 2, 0, max(0, w_img - side)))
    top = int(np.clip(cy - side / 2, 0, max(0, h_img - side)))
    crop = img[top : top + side, left : left + side]
    scale = crop_px / float(max(crop.shape[:2]))
    crop = cv2.resize(
        crop,
        (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )

    def to_px(px, py):
        return int(round((px - left) * scale)), int(round((py - top) * scale))

    lw = max(2, crop_px // 250)
    for contours, color, thickness in masks:
        for c in contours:
            poly = np.array([to_px(px, py) for px, py in c], dtype=np.int32)
            cv2.polylines(crop, [poly], True, color, thickness, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, label in enumerate(labels, start=1):
        if not visible[k]:
            continue
        color = _match_color(k)
        p = to_px(x[k], y[k])
        cv2.circle(crop, p, 4 * lw, (255, 255, 255), lw + 2, cv2.LINE_AA)
        cv2.circle(crop, p, 4 * lw, color, lw, cv2.LINE_AA)
        org = (p[0] + 5 * lw, p[1] - 3 * lw)
        cv2.putText(crop, label, org, font, 0.35 * lw, (255, 255, 255), 3 * lw,
                    cv2.LINE_AA)
        cv2.putText(crop, label, org, font, 0.35 * lw, color, lw, cv2.LINE_AA)
    if visible[0]:
        p = to_px(x[0], y[0])
        cv2.drawMarker(crop, p, (255, 255, 255), cv2.MARKER_CROSS, 8 * lw, lw + 3,
                       cv2.LINE_AA)
        cv2.drawMarker(crop, p, _MATCH_RANDOM_COLOR, cv2.MARKER_CROSS, 8 * lw,
                       lw + 1, cv2.LINE_AA)
    return crop


def _match_masks(view, result, color, lw):
    """Mask outlines to draw for a view: top mask, plus the 2nd if it decided."""
    contours = (view or {}).get("contours") or []
    masks = []
    if contours:
        masks.append((contours[0], color, lw))
    if len(contours) > 1 and result and not result[0] and len(result) > 1 and result[1]:
        masks.append((contours[1], _lighter(color), lw))
    return masks


def save_annotation_matches_to_pdf(
    records,
    output_filepath,
    review_all=False,
    crop_px=settings.MATCH_REVIEW_CROP_PX,
    jpeg_quality=70,
):
    """Save a review PDF of random-point -> manual annotation matches.

    One A4-landscape page per flagged match (every considered point with
    candidates when ``review_all``) from
    :meth:`substrata.annotations.Annotations.match_annotations`, ordered low,
    medium, unmatched, high. Each page has three panels:

    1. The random point's image with its SAM2 mask (red, like the cross).
    2. The assigned manual annotation's image (the nearest candidate's if
       unmatched) with its SAM2 mask in the candidate's colour.
    3. A conflicting candidate's image and mask (a non-assigned candidate with
       any in-mask agreement), else an overview of the image that sees the
       random point and the most (nearest) candidates, without masks.

    Candidates are numbered markers labelled with their id suffix, coloured
    consistently across panels and the candidate table. 2nd-ranked masks are
    drawn in a lighter shade when they decided a test. A wrong ``manual_id``
    can be corrected directly in the matched CSV.

    Args:
        records: Match records from ``match_annotations``.
        output_filepath: Output PDF path.
        review_all: Include every match, not only flagged ones.
        crop_px: Pixel size of each image panel (limits file size).
        jpeg_quality: JPEG quality of embedded images.
    """
    order = {"low": 0, "medium": 1, "": 2, "high": 3}
    pages = sorted(
        (
            rec
            for rec in records
            if rec["candidates"] and (rec["flagged"] or review_all)
        ),
        key=lambda rec: order.get(rec["manual_id_conf"], 2),
    )
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(False)
    margin = 10

    # Summary page
    counts = {}
    for rec in records:
        if rec["considered"]:
            key = rec["manual_id_conf"] or "unmatched"
            counts[key] = counts.get(key, 0) + 1
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.text(margin, margin + 5, "Annotation matches - review")
    pdf.set_font("Arial", size=10)
    lines = [f"{k}: {v}" for k, v in sorted(counts.items())] + [
        f"pages in this PDF: {len(pages)}"
        + (" (all matches)" if review_all else " (flagged only)")
        + ", ordered low, medium, unmatched"
        + (", high" if review_all else ""),
        "",
        "Red cross + red outline = random point and its SAM2 mask. Numbered "
        "markers = candidate manual annotations (id suffix), one colour per "
        "candidate; outlines in the same colour are that candidate's SAM2 mask.",
        "Lighter outline = 2nd-ranked SAM2 mask, drawn when it decided a test.",
        "fwd = random point inside the candidate's mask; rev = candidate inside "
        "the random point's mask; - = not visible in the image.",
        "high: one candidate agrees both ways (top masks). medium: agrees both "
        "ways using a 2nd mask, or one way only. low: several agree (nearest "
        "chosen) or nearest within the fallback distance.",
        "SA = mask area projected on the image plane (approximate).",
        "To correct a match, edit manual_id (and manual_id_conf) in the matched CSV.",
    ]
    for i, line in enumerate(lines):
        pdf.set_xy(margin, margin + 12 + 6 * i)
        pdf.multi_cell(277, 5, line)

    panel_w = (297 - 2 * margin - 2 * 4) / 3
    lw = max(2, crop_px // 250)
    for rec in tqdm(pages, desc="Review PDF", unit="page"):
        cands = rec["candidates"]
        ids = [c["id"] for c in cands]
        labels = _short_ids(ids)
        chosen = ids.index(rec["manual_id"]) + 1 if rec["manual_id"] else None
        shown = chosen or 1

        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(0, 0, 0)
        if chosen:
            conf = f", label conf {rec['manual_label_conf']}" if rec[
                "manual_label_conf"] else ""
            assigned = (
                f"{labels[chosen - 1]} ({rec['manual_label']}{conf})  -  "
                f"match: {rec['manual_id_conf']}"
            )
        else:
            assigned = "UNMATCHED"
        pdf.text(
            margin, margin + 4, f"Random {rec['id']} ({rec['label']}) -> {assigned}"
        )
        pdf.set_font("Arial", size=8)
        areas = (rec["view"] or {}).get("area_cm2") or []
        sa = ", ".join(
            f"{a:.0f} cm2 ({'top' if r == 0 else '2nd'} mask)"
            for r, a in enumerate(areas)
        )
        pdf.set_text_color(200, 0, 0)
        pdf.text(margin, margin + 9, f"Random colony SA ~ {sa or 'n/a'}")
        for k, c in enumerate(cands, start=1):
            b, g, r = _match_color(k)
            pdf.set_text_color(r, g, b)
            view_areas = (rec["manual_views"].get(c["id"]) or {}).get("area_cm2")
            m_sa = f"  SA ~ {view_areas[0]:.0f} cm2" if view_areas else ""
            conf = f" (conf {c['label_conf']})" if c["label_conf"] else ""
            pdf.text(
                margin,
                margin + 13 + 4 * (k - 1),
                f"{labels[k - 1]}: {c['id']}  {c['label']}{conf}  "
                f"dist {c['dist'] * 100:.1f} cm  fwd {_vote_str(c['fwd'])}  "
                f"rev {_vote_str(c['rev'])}{m_sa}"
                + ("   <- assigned" if k == chosen else ""),
            )
        pdf.set_text_color(0, 0, 0)
        top = margin + 15 + 4 * len(cands)

        # Panel 1: random point view with its own masks (red)
        panels = []
        view = rec["view"]
        if view is not None:
            rev_needed_2nd = [c["rev"] for c in cands]
            masks = _match_masks(view, None, _MATCH_RANDOM_COLOR, lw)
            if any(r and not r[0] and len(r) > 1 and r[1] for r in rev_needed_2nd):
                masks = _match_masks(view, (False, True), _MATCH_RANDOM_COLOR, lw)
            panels.append((f"random {rec['id']}", view["cam"], masks, [0, shown]))
        else:
            panels.append((f"random {rec['id']}", None, [], None))

        # Panel 2: assigned (or nearest) candidate's view
        c = cands[shown - 1]
        mview = rec["manual_views"].get(c["id"])
        title = f"{'assigned' if chosen else 'nearest'} {labels[shown - 1]}"
        panels.append(
            (title, mview["cam"] if mview else None,
             _match_masks(mview, c["fwd"], _match_color(shown), lw), [0, shown])
        )

        # Panel 3: conflicting candidate, else an overview without masks
        conflict = next(
            (
                k
                for k, c in enumerate(cands, start=1)
                if k != shown
                and any(v for v in (c["fwd"] or ()) + (c["rev"] or ()))
                and rec["manual_views"].get(c["id"]) is not None
            ),
            None,
        )
        if conflict:
            c = cands[conflict - 1]
            mview = rec["manual_views"][c["id"]]
            panels.append(
                (f"conflicting {labels[conflict - 1]}", mview["cam"],
                 _match_masks(mview, c["fwd"], _match_color(conflict), lw),
                 [0, conflict])
            )
        elif rec["overview"] is not None:
            cam, _, _, visible = rec["overview"]
            n = int(visible[1:].sum())
            panels.append((f"overview ({n}/{len(cands)} candidates)", cam, [], None))

        for p, (title, cam, masks, focus_ks) in enumerate(panels):
            x0 = margin + p * (panel_w + 4)
            pdf.text(x0, top, title)
            if cam is None:
                _draw_placeholder(pdf, x0, top + 2, panel_w, panel_w, "no view")
                continue
            img = _render_match_panel(cam, rec, labels, masks, focus_ks, crop_px)
            if img is None:
                _draw_placeholder(pdf, x0, top + 2, panel_w, panel_w, "no image")
                continue
            _, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            h_mm = panel_w * img.shape[0] / img.shape[1]
            pdf.image(BytesIO(buf.tobytes()), x=x0, y=top + 2, w=panel_w, h=h_mm)
            pdf.text(x0, top + 6 + h_mm, os.path.basename(cam.filepath))

    pdf.output(output_filepath)
    logger.info(f"Review PDF ({len(pages)} matches) saved to {output_filepath}")


def _ann_pcd_top_view(ann, radius=None, up_vector=None, rotation=0):
    """Return a PIL Image of ann.simple_pcd as a top-down OrthoMap, or None.

    Args:
        ann: Annotation with a ``simple_pcd`` attribute.
        radius: Sampling radius in metres; when set the highlight circle uses
            ``point_size_m`` with a red outline and transparent fill.
        up_vector: Up vector for the OrthoMap projection.  Defaults to
            ``[0, 0, 1]`` when ``None``.
        rotation: Clockwise rotation in degrees applied to the output image.
    """
    try:
        from substrata.ortho import OrthoMap
        from PIL import ImageFilter

        if ann.simple_pcd is None:
            return None
        view = OrthoMap(ann.simple_pcd, up_vector=up_vector, display_rotation=rotation)
        kwargs = dict(highlights=ann, point_size=15)
        if radius is not None:
            kwargs.update(
                point_size_m=radius * 2,
                point_color=None,
                point_outline=(255, 0, 0),
            )
        img = view.show(**kwargs)
        n_pts = max(len(ann.simple_pcd.points), 1)
        raw = int(np.sqrt(view.width * view.height / n_pts) * 2.5)
        filter_size = max(3, raw if raw % 2 == 1 else raw + 1)
        return img.filter(ImageFilter.MinFilter(size=filter_size))
    except Exception as e:
        logger.warning("_ann_pcd_top_view failed for %s: %s", ann.id, e)
        return None


def plot_measurements(
    annotations,
    measurements: Optional[List[str]] = None,
    cols: int = 4,
    width: int = 16,
    height_per_row: int = 4,
    title: Optional[str] = None,
) -> Any:
    """Box plots of numeric measurements grouped by Annotation.label.

    Args:
        annotations: Annotations container whose items each have a ``label``
            attribute and a ``measurements`` dict.
        measurements: Optional list of measurement keys to include. When None,
            all numeric (scalar) keys are auto-detected (``_image`` keys are
            always excluded).
        cols: Maximum number of subplots per row.
        width: Figure width in inches.
        height_per_row: Height in inches per subplot row.
        title: Optional overall figure title.

    Returns:
        matplotlib Figure.
    """
    import math

    # --- 1. Collect numeric measurement keys ---
    all_keys: set = set()
    for ann in annotations.data.values():
        for k, v in ann.measurements.items():
            if "_image" in k:
                continue
            if np.isscalar(v) and not isinstance(v, (bool, str)):
                all_keys.add(k)

    if measurements is not None:
        measurement_keys = [k for k in measurements if k in all_keys]
    else:
        measurement_keys = sorted(all_keys)

    if not measurement_keys:
        raise ValueError("No numeric measurements found in the provided Annotations.")

    # --- 2. Collect values grouped by label ---
    unique_labels = sorted(
        {
            getattr(ann, "label", None)
            for ann in annotations.data.values()
            if getattr(ann, "label", None) is not None
        }
    )

    data: Dict[str, Dict[str, list]] = {
        key: {label: [] for label in unique_labels} for key in measurement_keys
    }
    for ann in annotations.data.values():
        label = getattr(ann, "label", None)
        if label is None:
            continue
        for key in measurement_keys:
            v = ann.measurements.get(key)
            if v is not None and np.isscalar(v) and not isinstance(v, (bool, str)):
                data[key][label].append(float(v))

    # --- 3. Build subplot grid ---
    n_plots = len(measurement_keys)
    n_cols = min(cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height_per_row * n_rows), squeeze=False
    )

    # Assign a distinct color per label
    cmap_labels = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20
    label_colors = {
        label: cmap_labels(i / max(len(unique_labels) - 1, 1))
        for i, label in enumerate(unique_labels)
    }

    rng = np.random.default_rng(0)

    # --- 4. Draw box plots ---
    for idx, key in enumerate(measurement_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        plot_data = [data[key][label] for label in unique_labels]
        bp = ax.boxplot(
            plot_data,
            labels=unique_labels,
            sym="",                 # suppress default fliers; we draw jitter instead
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )

        for i, (label, box) in enumerate(zip(unique_labels, bp["boxes"])):
            color = label_colors[label]
            box.set(facecolor=(*color[:3], 0.25), edgecolor=color)
            bp["whiskers"][2 * i].set_color(color)
            bp["whiskers"][2 * i + 1].set_color(color)
            bp["caps"][2 * i].set_color(color)
            bp["caps"][2 * i + 1].set_color(color)

            vals = plot_data[i]
            n = len(vals)
            if n > 0:
                jitter = rng.uniform(-0.2, 0.2, size=n)
                ax.scatter(
                    np.full(n, i + 1) + jitter,
                    vals,
                    color=color,
                    s=15,
                    alpha=0.6,
                    edgecolors="none",
                    zorder=3,
                )
        tick_labels = [
            f"{label} (n={len(plot_data[i])})"
            for i, label in enumerate(unique_labels)
        ]
        # Fully vertical (90°) x-tick labels so class names stay legible when
        # there are many of them.
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", va="top")
        ax.set_title(key, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_rasterized(True)

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def save_measurement_visualizations_to_pdf(
    annotations,
    output_filepath,
    orthomap=None,
    radius: float = None,
):
    """Save per-annotation measurement visualizations to a landscape A4 PDF.

    Each annotation occupies one page with three rows:

    * **Row 1** – OrthoMap overview (left ¾) + annotation metadata text (right ¼).
    * **Row 2** – Full camera image with annotation marked | zoomed crop |
      point-cloud rendered from the camera direction.
    * **Row 3** – Measurement images (one column each) + scalar values column.

    Args:
        annotations: Annotations object whose items will be rendered.
        output_filepath: Destination PDF path.
        orthomap: Optional shared :class:`~substrata.ortho.OrthoMap`.  When
            provided the annotation location is highlighted on the map in row 1.
        radius: Sampling radius in metres.  When provided, the OrthoMap
            highlight circle diameter equals ``radius * 2`` metres and the
            zoomed-crop column 2 is masked to the same radius (greying out
            pixels beyond it).
    """
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(False)
    pdf.set_font("Arial", size=7)

    margin = 10
    # A4 landscape: 297 × 210 mm  →  usable 277 × 190 mm
    usable_w = 297 - 2 * margin
    usable_h = 210 - 2 * margin
    row_h = usable_h / 3
    label_gap = 3  # mm below an image before the text label
    label_h = 4  # mm for a single text row

    standard_order = [
        "gapF_image",
        "elevation_image",
        "roughness_image",
        "vector_dispersion_image",
    ]

    for ann in annotations.data.values():
        pdf.add_page(orientation="L")
        pdf.set_font("Arial", size=7)
        pdf.set_text_color(0, 0, 0)

        # Use per-annotation radius when no global radius was supplied.
        eff_radius = radius if radius is not None else getattr(ann, "radius", None)

        # ── Row 1: OrthoMap + metadata ────────────────────────────────────
        r1_y = margin
        ortho_w = usable_w * 3 / 4
        text_x = margin + ortho_w
        text_w = usable_w / 4

        if orthomap is not None:
            try:
                ortho_kwargs = dict(highlights=ann, width=1200, height=600)
                if eff_radius is not None:
                    ortho_kwargs.update(
                        point_size_m=eff_radius * 2,
                        point_color=None,
                        point_outline=(255, 0, 0),
                    )
                ortho_img = orthomap.show(**ortho_kwargs)
                _pil_to_pdf(pdf, ortho_img, margin, r1_y, ortho_w, row_h)
            except Exception as e:
                logger.warning("OrthoMap render failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, margin, r1_y, ortho_w, row_h, "OrthoMap error")
        else:
            _draw_placeholder(pdf, margin, r1_y, ortho_w, row_h, "No OrthoMap")

        # Metadata text block
        try:
            depth_str = f"{ann.depth_in_m:.4f} m"
        except Exception:
            depth_str = "N/A"

        coords_str = (
            f"[{ann.coords[0]:.4f}, {ann.coords[1]:.4f}, {ann.coords[2]:.4f}]"
            if ann.coords is not None
            else "N/A"
        )
        orig_coords_str = (
            f"[{ann.orig_coords[0]:.4f}, {ann.orig_coords[1]:.4f}, {ann.orig_coords[2]:.4f}]"
            if getattr(ann, "orig_coords", None) is not None
            else "N/A"
        )

        try:
            n_pts = (
                len(ann.simple_pcd.points)
                if getattr(ann, "simple_pcd", None) is not None
                else "N/A"
            )
        except Exception:
            n_pts = "N/A"

        meta_lines = [
            ("ID", str(ann.id) if ann.id else "N/A"),
            ("Label", str(ann.label) if getattr(ann, "label", None) else "N/A"),
            ("Coords", coords_str),
            ("Orig coords", orig_coords_str),
            ("Depth", depth_str),
            ("Subsampled pts", str(n_pts)),
        ]
        line_h = 6
        pdf.set_font("Arial", "B", size=7)
        pdf.set_xy(text_x + 2, r1_y + 3)
        pdf.cell(text_w - 4, line_h, "Annotation info", ln=1)
        pdf.set_font("Arial", size=7)
        for key, val in meta_lines:
            pdf.set_xy(text_x + 2, pdf.get_y())
            pdf.multi_cell(text_w - 4, line_h * 0.8, f"{key}: {val}", border=0)

        # ── Row 2: Camera views ───────────────────────────────────────────
        r2_y = margin + row_h
        col_w = usable_w / 3
        img_h2 = row_h - label_gap - label_h

        ann_has_image_match = (
            getattr(ann, "image_match", None) is not None
            and getattr(ann.image_match, "filepath", None) is not None
        )

        # Col 1: full camera image, oriented + square-cropped, annotation marked
        col1_x = margin
        if ann_has_image_match:
            try:
                highlight_radius = 50
                if eff_radius is not None:
                    try:
                        ppm = ann.image_match.pixels_per_mm
                        if ppm is not None:
                            highlight_radius = max(1, int(eff_radius * 1000.0 * ppm))
                    except Exception:
                        pass
                full_img = ann.image_match.cam.render(
                    highlight_pixels=[ann.image_match.x, ann.image_match.y],
                    orient=True,
                    square=True,
                    highlight_radius=highlight_radius,
                )
                if full_img is not None:
                    _pil_to_pdf(pdf, full_img, col1_x, r2_y, col_w, img_h2)
                else:
                    _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "Image error")
            except Exception as e:
                logger.warning("Full image render failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "Image error")
        else:
            _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "No image match")
        pdf.set_xy(col1_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "full image", ln=0)

        # Col 2: zoomed crop via ImageMatch.render()
        # If a sampling radius is provided and no circular mask has been set yet,
        # create one so that render() can auto-apply the grey-out.
        col2_x = margin + col_w
        if ann_has_image_match:
            try:
                if eff_radius is not None:
                    existing = getattr(ann.image_match, "mask", None)
                    already_circular = (
                        existing is not None
                        and hasattr(existing, "radius_m")
                        and existing.radius_m is not None
                    )
                    if not already_circular:
                        try:
                            ann.image_match.create_circular_masks([eff_radius])
                        except Exception:
                            pass
                crop_pil = ann.image_match.render(crop_w=1000, crop_h=1000, orient=True)
                _pil_to_pdf(pdf, crop_pil, col2_x, r2_y, col_w, img_h2)
            except Exception as e:
                logger.warning("Zoomed crop failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, col2_x, r2_y, col_w, img_h2, "Crop error")
        else:
            _draw_placeholder(pdf, col2_x, r2_y, col_w, img_h2, "No image match")
        pdf.set_xy(col2_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "zoomed crop", ln=0)

        # Col 3: top-down point cloud view
        col3_x = margin + 2 * col_w
        up_vec = getattr(orthomap, "_up_vector", None) if orthomap is not None else None
        pcd_img = _ann_pcd_top_view(ann, radius=eff_radius, up_vector=up_vec, rotation=90)
        if pcd_img is not None:
            _pil_to_pdf(pdf, pcd_img, col3_x, r2_y, col_w, img_h2)
        else:
            _draw_placeholder(pdf, col3_x, r2_y, col_w, img_h2, "No point cloud")
        pdf.set_xy(col3_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "point cloud view", ln=0)

        # ── Row 3: Measurements ───────────────────────────────────────────
        r3_y = margin + 2 * row_h

        meas = getattr(ann, "measurements", {}) or {}
        image_keys_all = [k for k, v in meas.items() if "_image" in k and v is not None]
        ordered_img_keys = [k for k in standard_order if k in image_keys_all]
        ordered_img_keys += sorted(k for k in image_keys_all if k not in standard_order)

        scalar_meas = {
            k: v
            for k, v in meas.items()
            if "_image" not in k
            and v is not None
            and isinstance(v, (int, float, np.integer, np.floating))
        }
        extra = getattr(ann, "extra_coords", None) or {}

        has_images = bool(ordered_img_keys)
        has_scalars = bool(scalar_meas or extra)

        if has_images or has_scalars:
            n_img_cols = len(ordered_img_keys)
            n_r3_cols = n_img_cols + 1  # +1 for scalar column
            cell_w3 = usable_w / n_r3_cols
            img_h3 = row_h - label_gap - label_h

            for ci, img_key in enumerate(ordered_img_keys):
                cx = margin + ci * cell_w3
                img_val = meas[img_key]
                try:
                    if isinstance(img_val, np.ndarray):
                        arr = (
                            img_val
                            if img_val.dtype == np.uint8
                            else (
                                (np.clip(img_val, 0, 1) * 255).astype(np.uint8)
                                if img_val.max() <= 1.0
                                else img_val.astype(np.uint8)
                            )
                        )
                        pil_v = Image.fromarray(arr)
                    elif isinstance(img_val, Image.Image):
                        pil_v = img_val
                    elif hasattr(img_val, "to_image"):
                        raw = img_val.to_image(format="png", width=600, height=400)
                        pil_v = Image.open(BytesIO(raw)).convert("RGB")
                    else:
                        pil_v = Image.fromarray(np.array(img_val))
                    if img_key == "gapF_image":
                        arr_rgb = np.array(pil_v.convert("RGB"))
                        arr_rgb[np.all(arr_rgb == 0, axis=-1)] = 255
                        pil_v = Image.fromarray(arr_rgb).rotate(90, expand=True)
                    _pil_to_pdf(pdf, pil_v, cx, r3_y, cell_w3, img_h3)
                except Exception as e:
                    logger.warning(
                        "Meas image %s failed for %s: %s", img_key, ann.id, e
                    )
                    _draw_placeholder(pdf, cx, r3_y, cell_w3, img_h3)
                pdf.set_xy(cx + 2, r3_y + img_h3 + label_gap)
                pdf.cell(cell_w3 - 4, label_h, img_key, ln=0)

            # Scalar text column (always last)
            sc_x = margin + n_img_cols * cell_w3
            pdf.set_fill_color(245, 245, 245)
            pdf.rect(sc_x, r3_y, cell_w3, row_h, style="F")
            pdf.set_fill_color(0, 0, 0)
            pdf.set_font("Arial", "B", size=6)
            pdf.set_xy(sc_x + 2, r3_y + 2)
            pdf.cell(cell_w3 - 4, 4, "Measurements", ln=1)
            pdf.set_font("Arial", size=6)
            for k, v in scalar_meas.items():
                if pdf.get_y() > r3_y + row_h - 4:
                    break
                pdf.set_xy(sc_x + 2, pdf.get_y())
                val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
                pdf.multi_cell(cell_w3 - 4, 3.5, f"{k}: {val_str}", border=0)
            if extra:
                pdf.set_font("Arial", "B", size=6)
                pdf.set_xy(sc_x + 2, pdf.get_y())
                pdf.cell(cell_w3 - 4, 4, "Extra coords", ln=1)
                pdf.set_font("Arial", size=6)
                for k, v in extra.items():
                    if pdf.get_y() > r3_y + row_h - 4:
                        break
                    pdf.set_xy(sc_x + 2, pdf.get_y())
                    pdf.multi_cell(cell_w3 - 4, 3.5, f"{k}: {v}", border=0)
            pdf.set_font("Arial", size=7)

    pdf.output(output_filepath)
    logger.info("PDF created: %s", output_filepath)


def _scalebar_summary(scalebars):
    return (
        f"Number of scalebars: {scalebars.scalebars}\n"
        f"Scale factor: {scalebars.scalefactor:.5f}\n"
        f"Variance: {scalebars.var:.10f}\n"
        f"Std Error: {scalebars.sterr:.10f}"
    )


def scalebar_figures(scalebars, pcd):
    """Two top-down figures (one per target) for each located scale bar.

    Args:
        scalebars: :class:`~substrata.annotations.Scalebars`.
        pcd: The point cloud the targets were located in.

    Returns:
        list: matplotlib figures.
    """
    figs = []
    for scalebar in scalebars.data:
        if scalebar.target1_coords is None or scalebar.target2_coords is None:
            continue
        radius = float(scalebar.length) * 5
        for label, coords in (
            (scalebar.target1_label, scalebar.target1_coords),
            (scalebar.target2_label, scalebar.target2_coords),
        ):
            target = pcd.subsample_pointcloud_by_radius(coords, radius)
            figs.append(plot_2d(
                target, width=4, height=4, highlight_coords=coords,
                title=label + "\n" + str(coords),
            ))
    return figs


def show_scalebars(scalebars, pcd):
    """Print the scale-bar summary and show the target figures (``Scalebars.show``)."""
    if scalebars.scalebars is None:
        return []
    print(_scalebar_summary(scalebars))
    figs = scalebar_figures(scalebars, pcd)
    for fig in figs:
        fig.show()
    return figs


def save_scalebars_pdf(scalebars, pcd, filepath=None):
    """Save the scale-bar summary and target figures to a PDF (``Scalebars.save_pdf``).

    Each figure is closed once saved (none is shown, e.g. in a notebook); the
    matplotlib backend is left as it is. Default path:
    ``<cloud base>_scalebars.pdf``.
    """
    if scalebars.scalebars is None:
        return
    from matplotlib.backends.backend_pdf import PdfPages

    if filepath is None:
        filepath = f"{os.path.splitext(pcd.filepath)[0]}_scalebars.pdf"
    with PdfPages(filepath) as pdf:
        figs = [plot_text(_scalebar_summary(scalebars))]
        try:
            figs += scalebar_figures(scalebars, pcd)
            for fig in figs:
                pdf.savefig(fig)
        finally:
            for fig in figs:
                plt.close(fig)


def show_measurement_image(annotation, key, width=12):
    """Display a stored measurement image of an annotation (e.g. ``tpi_image``).

    Args:
        annotation: The annotation (its ``measurements`` dict).
        key: Measurement key holding an (H, W, 3) image array.
        width: Figure width in inches (height scales to aspect ratio).
    """
    img = annotation.measurements.get(key)
    if img is None:
        logger.warning("No image found for '%s' on annotation %s", key,
                       annotation.id)
        return
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{annotation.id} — {key}", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_measurement_images(annotation, width=12):
    """Display all stored measurement images (keys ending in ``_image``)."""
    image_keys = [
        k for k, v in annotation.measurements.items()
        if k.endswith("_image") and v is not None
    ]
    if not image_keys:
        logger.warning("No measurement images found on annotation %s",
                       annotation.id)
        return
    for key in image_keys:
        show_measurement_image(annotation, key, width=width)
