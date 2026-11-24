"""Reports of CLI runs.

Training-crop contact sheets and the training summary (``substrata train``,
counts from :mod:`substrata.classification`), classifier evaluation stats and
example pages, the ``substrata imagesync`` report, the ``substrata camsync``
summary and sanity PDF, and the curation review sheets (``substrata curate
--sheets``). The data are loaded and computed by the calling modules
(``cli``, ``classification``, ``imagesync``, ``curation``); these functions only
lay them out.
"""

# Standard Library
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Sequence

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from PIL import Image, ImageDraw
from tqdm import tqdm

# Local Modules
from substrata import settings
from substrata.logging import logger
from substrata.visualizations._common import _draw_placeholder


def save_training_crops_pdf(
    groups,
    output_filepath,
    n_cols=settings.TRAIN_PDF_COLS,
    n_rows=settings.TRAIN_PDF_ROWS,
    thumb_px=settings.TRAIN_PDF_THUMB_PX,
    crop_dirs=settings.TRAIN_CROP_DIRS,
    crop_px=None,
):
    """Save a contact sheet of training crops, one or more pages per class.

    Each crop is captioned with its annotation id (searchable PDF text), plus
    its raw label when that was collapsed into the class and a ``val``/``test``
    tag for validation/test crops. Crops are downscaled to ``thumb_px`` and
    stored as JPEG to keep the PDF small.

    Args:
        groups: ``class -> (n_crops, [(path, id, raw_label, split_dir), ...])``
            (see :func:`substrata.classification.training_crops_by_class`).
        output_filepath: Destination PDF path.
        n_cols: Crops per row.
        n_rows: Rows per page.
        thumb_px: Longest side of the embedded crops in pixels.
        crop_dirs: Train/validation/test folder names (for the split tags).
        crop_px: Crop size in pixels, shown in the page headers (all crops are
            drawn at the same size on the page).

    Returns:
        Number of pages written.
    """
    split_tags = {crop_dirs[1]: "val", crop_dirs[2]: "test"}
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(False)
    margin, header_h, caption_h = 8.0, 7.0, 2.8
    cell_w = (pdf.w - 2 * margin) / n_cols
    cell_h = (pdf.h - 2 * margin - header_h) / n_rows
    side = min(cell_w - 1.0, cell_h - caption_h - 0.4)
    per_page = n_cols * n_rows

    def _caption(text, x, y, max_size=3.5):
        size = max_size
        pdf.set_font("Helvetica", size=size)
        while size > 1.5 and pdf.get_string_width(text) > cell_w - 0.6:
            size -= 0.25
            pdf.set_font("Helvetica", size=size)
        pdf.text(x, y, text)

    n_total = sum(len(items) for _, items in groups.values())
    with tqdm(total=n_total, desc="Training crops PDF", unit="crop") as bar:
        for cls, (n_crops, items) in groups.items():
            raw_counts = {}
            for _, _, raw, _ in items:
                raw_counts[raw] = raw_counts.get(raw, 0) + 1
            raw_text = ", ".join(f"{r} {n}" for r, n in sorted(raw_counts.items()))
            n_pages = max(1, -(-len(items) // per_page))
            for page in range(n_pages):
                pdf.add_page()
                pdf.set_font("Helvetica", style="B", size=10)
                title = f"{cls} - showing {len(items)} of {n_crops} crops"
                if crop_px:
                    title += f" ({crop_px} px)"
                if n_pages > 1:
                    title += f" (page {page + 1}/{n_pages})"
                pdf.text(margin, margin + 3.5, title)
                pdf.set_font("Helvetica", size=6)
                pdf.text(margin, margin + 6.2, f"raw labels shown: {raw_text}")
                page_items = items[page * per_page:(page + 1) * per_page]
                for i, (path, ann_id, raw, split_dir) in enumerate(page_items):
                    x = margin + (i % n_cols) * cell_w
                    y = margin + header_h + (i // n_cols) * cell_h
                    img_x = x + (cell_w - side) / 2.0
                    try:
                        with Image.open(path) as im:
                            im = im.convert("RGB")
                            im.thumbnail((thumb_px, thumb_px))
                            buf = BytesIO()
                            im.save(buf, format="JPEG", quality=85)
                        buf.seek(0)
                        pdf.image(buf, x=img_x, y=y, w=side, h=side)
                    except Exception as e:  # noqa: BLE001 - unreadable crop
                        logger.warning("Unreadable crop %s: %s", path, e)
                        pdf.set_font("Helvetica", size=4)
                        _draw_placeholder(pdf, img_x, y, side, side, "unreadable")
                    _caption(ann_id, img_x, y + side + 1.2)
                    extra = [t for t in (
                        raw if raw != cls else "", split_tags.get(split_dir, "")
                    ) if t]
                    if extra:
                        _caption(" ".join(extra), img_x, y + side + 2.6, 3.0)
                    bar.update(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    pdf.output(output_filepath)
    return pdf.pages_count


def _thumb(path, size):
    from PIL import Image

    im = Image.open(path)
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    im.thumbnail((size, size))
    return im


def write_imagesync_report_pdf(report: Dict[str, Any], targets: Sequence[Any], path: str, *,
                               title: str = "", max_posed: int = 9, per_page: int = 3) -> str:
    """The run's quality check: a summary page with the offset's scores, then close-ups next to their frame (the
    close-up's centre mapped into the frame by the matches' homography: red; where its pose aims: green). Every
    close-up that was not posed is shown, and up to ``max_posed`` posed ones (spread over the run).

    ``report`` is :func:`substrata.imagesync`'s run report (``offset``, ``status``, ``matches``, ``poses``, ...);
    ``imagesync.write_report_pdf`` delegates here."""
    # (Figure objects, no pyplot: the caller's matplotlib backend is left as it is)
    from substrata.imagesync import map_point
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    off = report["offset"]
    status = report["status"]
    counts: Dict[str, int] = {}
    for s, _n in status.values():
        counts[s] = counts.get(s, 0) + 1
    with PdfPages(path) as pdf:
        fig = Figure(figsize=(11.7, 8.3))
        lines = [title or "camsync --match-images", "",
                 f"Time offset: {report['time_offset']} s (a close-up taken at t is in the frame of t + offset)"
                 if report["time_offset"] is not None else f"No time offset: {off.get('reason', '')}",
                 f"  from the images: peak {off.get('offset')} s, {off.get('prominence', 0):.1f}x the median vote, "
                 f"{off.get('n_agree', 0)} of {off.get('n_targets', 0)} close-ups agree; offsets tried "
                 f"{off.get('window')}",
                 "  confirmed with the learned matcher: " + ", ".join(
                     f"{r['matched']} of {r['sampled']} close-ups match at {r['offset']} s"
                     for r in (report.get("confirm") or {}).get("tried", [])) if (report.get("confirm") or {}).get("tried")
                 else "",
                 f"  close-up centre in a frame ~{tuple(int(v) for v in report['aim'])} px, ~{report['width_px']:.0f} px "
                 f"wide, turned {off.get('rotation_deg') or 0:.0f} deg (matched turned {90 * report.get('turns', 0)} deg)"
                 if report.get("aim") else "",
                 "  unaligned frames posed by substrata (--pose-frames): {} of {}".format(
                     sum(bool(r.get("used")) for r in report["frame_poses"].values()), len(report["frame_poses"]))
                 if report.get("frame_poses") else "",
                 "  rig (close-up relative to its frame) from {n} posed close-ups: spread {a:.1f} cm, {b:.1f} deg; "
                 "close-ups without a good match estimated from it".format(
                     n=report["rig"]["n"], a=100 * report["rig"]["spread_m"], b=report["rig"]["spread_deg"])
                 if report.get("rig") else "", "",
                 "Close-ups: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]
        rng = [p["range"] for cid, p in report["poses"].items() if p and status.get(cid, ("",))[0] == "posed"]
        if rng:
            lines.append(f"  posed: camera to reef {np.median(rng):.2f} m (median; {min(rng):.2f}-{max(rng):.2f})")
        fig.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
        if off.get("ks") is not None and len(off["ks"]):
            ax = fig.add_axes([0.08, 0.08, 0.86, 0.45])
            ax.plot(off["ks"], off["curve"], lw=0.8)
            if off.get("offset") is not None:
                ax.axvline(off["offset"], color="red", lw=0.8, ls="--")
            ax.set_xlabel("offset (s): frame time - close-up time")
            ax.set_ylabel("votes (each close-up's matches / its best)")
        pdf.savefig(fig)

        rows = sorted(targets, key=lambda c: c.filename)
        posed = [c for c in rows if status.get(str(c.cam_id), ("",))[0] == "posed"]
        keep = {str(c.cam_id) for c in rows if status.get(str(c.cam_id), ("",))[0] != "posed"}
        if rng:  # (and the posed ones at an unusual range)
            med = float(np.median(rng))
            keep |= {str(c.cam_id) for c in posed if (report["poses"].get(str(c.cam_id)) or {}).get("range")
                     and not (med / 1.5 <= report["poses"][str(c.cam_id)]["range"] <= 1.5 * med)}
        if posed and max_posed > 0:
            keep |= {str(posed[i].cam_id) for i in np.linspace(0, len(posed) - 1, min(max_posed, len(posed))).round()
                     .astype(int)}
        show = [c for c in rows if str(c.cam_id) in keep]
        for start in range(0, len(show), per_page):
            fig = Figure(figsize=(8.3, 11.7))
            axes = fig.subplots(per_page, 2, squeeze=False)
            for r, c in enumerate(show[start:start + per_page]):
                cid = str(c.cam_id)
                s, note = status.get(cid, ("no pose", ""))
                m = report["matches"].get(cid) or {}
                p = report["poses"].get(cid) or {}
                a0, a1 = axes[r]
                a0.imshow(_thumb(c.filepath, 700))
                head = f"{c.filename}: {s}" + (f" ({note})" if note else "")
                info = f"{m.get('n', 0)} matches"
                if p and s != "estimated":
                    info += f", {p['inliers']} pose inliers, error {p['error']:.0f} px" + (
                        f", range {p['range']:.2f} m" if p.get("range") else "")
                elif p:
                    info += f"; estimated: range {p['range']:.2f} m" if p.get("range") else "; estimated"
                a0.set_title(head + "\n" + info, fontsize=7, loc="left")
                src = m.get("source")
                if s == "estimated" and p.get("frame") is not None and p.get("aim") is not None:
                    src = p["frame"]
                    x, y, *_ = src.get_pixel_coords(p["aim"])
                    big = _thumb(src.filepath, max(src.sensor.width, src.sensor.height))
                    if x is not None:
                        half = 1.5 * max(report.get("width_px") or 400, 200)
                        box = (int(x - half), int(y - 0.6 * half), int(x + half), int(y + 0.6 * half))
                        a1.imshow(big.crop(box), extent=(box[0], box[2], box[3], box[1]))
                        a1.plot(x, y, "x", color="lime", ms=10, mew=2)
                    else:
                        a1.imshow(big)
                    a1.set_title(f"{src.filename} ({p.get('dt', 0):+d} s from the offset time): where the estimate "
                                 "aims (green)", fontsize=7, loc="left")
                elif src is not None and m.get("H") is not None:
                    hc = map_point(m["H"], (c.sensor.width / 2, c.sensor.height / 2))
                    big = _thumb(src.filepath, max(src.sensor.width, src.sensor.height))
                    half = 1.5 * max(report.get("width_px") or 400, 200)
                    box = (int(hc[0] - half), int(hc[1] - 0.6 * half), int(hc[0] + half), int(hc[1] + 0.6 * half))
                    a1.imshow(big.crop(box), extent=(box[0], box[2], box[3], box[1]))
                    a1.plot(*hc, "+", color="red", ms=14, mew=2)
                    if p and p.get("aim") is not None:
                        x, y, *_ = src.get_pixel_coords(p["aim"])
                        if x is not None:
                            a1.plot(x, y, "x", color="lime", ms=10, mew=2)
                    a1.set_title(f"{src.filename} ({m.get('dt', 0):+d} s from the offset time)", fontsize=7, loc="left")
                for a in (a0, a1):
                    a.set_xticks([])
                    a.set_yticks([])
            for r in range(len(show[start:start + per_page]), per_page):
                axes[r][0].axis("off")
                axes[r][1].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
    return path


def save_training_summary_pdf(
    pdf_path, info, counts_by_split,
    crop_dirs=settings.TRAIN_CROP_DIRS,
    metrics=None,
):
    """Render a training-run summary to a multi-page PDF.

    Page 1 is a text summary of the run settings, the per-split crop totals and
    (if given) the final metrics. The remaining pages show a horizontal stacked
    bar chart of per-training-class crop counts (train/validation/test),
    paginated ``settings.TRAIN_SUMMARY_CLASSES_PER_PAGE`` classes at a time and
    sorted by total count.

    Args:
        pdf_path: Destination PDF path.
        info: Ordered ``(label, value)`` rows of run settings to print.
        counts_by_split: ``split_dir -> Counter(class -> count)`` (see
            :func:`substrata.classification.count_crops_by_class`).
        crop_dirs: Folder names for the three splits (train/val/test order).
        metrics: Optional ``{metric_name: value}`` from the final epoch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    train_name, valid_name, test_name = crop_dirs
    classes = sorted(
        set().union(*[set(c) for c in counts_by_split.values()])
        if counts_by_split else set(),
        key=lambda c: (
            -sum(counts_by_split.get(s, {}).get(c, 0) for s in crop_dirs), c
        ),
    )

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Page 1: settings + totals + final metrics.
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle("Training summary", fontsize=12, y=0.98)
        lines = [f"{k}: {v}" for k, v in info]
        lines.append("")
        lines.append("Crops per split:")
        for split in crop_dirs:
            tot = sum(counts_by_split.get(split, {}).values())
            lines.append(f"  {split}: {tot}")
        grand_total = sum(
            sum(counts_by_split.get(s, {}).values()) for s in crop_dirs
        )
        lines.append(f"  total: {grand_total}")
        if metrics:
            lines.append("")
            lines.append("Final-epoch metrics:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")
        ax.text(
            0.02, 0.96, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Remaining pages: per-class stacked bar chart of crop counts.
        per_page = settings.TRAIN_SUMMARY_CLASSES_PER_PAGE
        for start in range(0, len(classes), per_page):
            page = classes[start:start + per_page]
            tr = np.array([counts_by_split.get(train_name, {}).get(c, 0) for c in page])
            va = np.array([counts_by_split.get(valid_name, {}).get(c, 0) for c in page])
            te = np.array([counts_by_split.get(test_name, {}).get(c, 0) for c in page])
            y = np.arange(len(page))
            fig, ax = plt.subplots(figsize=(8.5, max(3, len(page) * 0.25 + 1)))
            ax.barh(y, tr, color="#4c72b0", label=train_name)
            ax.barh(y, va, left=tr, color="#dd8452", label=valid_name)
            ax.barh(y, te, left=tr + va, color="#55a868", label=test_name)
            ax.set_yticks(y)
            ax.set_yticklabels(page, fontsize=6)
            ax.invert_yaxis()  # largest class on top
            ax.set_xlabel("Crops")
            ax.set_title("Crops per training class")
            ax.legend(fontsize=7, loc="lower right")
            totals = tr + va + te
            for i, t in enumerate(totals):
                ax.text(t, i, f" {int(t)}", va="center", fontsize=6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logger.info("Wrote training summary PDF: %s", pdf_path)


def _classifier_example_pages(pdf, labels, examples):
    """Append example-classification pages to an open ``PdfPages``.

    One row per category (in ``labels`` order, matching the confusion matrix),
    with up to ``settings.TRAIN_EXAMPLES_PER_CLASS`` example crops as columns.
    Each crop's title is the predicted label; misclassified crops get a red
    title and a red border. Rows are paginated
    ``settings.TRAIN_EXAMPLE_ROWS_PER_PAGE`` categories at a time.
    """
    per_class = settings.TRAIN_EXAMPLES_PER_CLASS
    rows_per_page = settings.TRAIN_EXAMPLE_ROWS_PER_PAGE

    # Keep confusion-matrix order; skip labels with no example crops.
    cats = [lab for lab in labels if examples.get(lab)]
    if not cats:
        return

    for start in range(0, len(cats), rows_per_page):
        page_cats = cats[start:start + rows_per_page]
        fig, axes = plt.subplots(
            len(page_cats), per_class,
            figsize=(per_class * 1.1 + 1.0, len(page_cats) * 1.25),
            squeeze=False,
        )
        fig.suptitle(
            "Example classifications (title = prediction; red = misclassified)",
            fontsize=11, y=0.99,
        )
        for r, cat in enumerate(page_cats):
            samples = examples.get(cat, [])[:per_class]
            for c in range(per_class):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(
                        cat, rotation=0, ha="right", va="center", fontsize=7,
                    )
                if c >= len(samples):
                    ax.axis("off")
                    continue
                fp, pred = samples[c]
                try:
                    im = np.asarray(Image.open(fp).convert("RGB"))
                    ax.imshow(im)
                except OSError:
                    ax.text(
                        0.5, 0.5, "(unreadable)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=5,
                    )
                wrong = pred != cat
                ax.set_title(
                    pred, fontsize=5, color="red" if wrong else "black",
                )
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red" if wrong else "0.6")
                    spine.set_linewidth(2.0 if wrong else 0.5)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)


def save_classifier_stats_pdf(
    pdf_path, split_folder, n, acc, report, cm, labels, examples=None,
):
    """Render the evaluation stats to a multi-page PDF.

    Page 1 is a text summary (accuracy + per-class precision/recall/F1); page 2
    is the confusion-matrix heatmap, row-normalised so per-class structure stays
    visible despite large differences in class size; the remaining pages show
    example classified crops per category (see :func:`_classifier_example_pages`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Page 1: text summary.
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle(f"Classifier stats - {split_folder}", fontsize=12, y=0.98)
        text = (
            f"Crops evaluated: {n}\n"
            f"Overall accuracy: {acc:.4f}\n\n"
            f"{report}"
        )
        ax.text(
            0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: confusion matrix heatmap. Colour by row-normalised values
        # (fraction of each true class) so the diagonal and per-class errors
        # stay visible regardless of how many images back each class; a few
        # huge classes would otherwise saturate a raw-count colour scale and
        # wash out every other cell.
        cm = np.asarray(cm)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm, row_sums, out=np.zeros(cm.shape, dtype=float),
            where=row_sums != 0,
        )

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5),
                                        max(5, len(labels) * 0.5)))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        # Keep absolute counts in view via the y-axis (n = images per class).
        ax.set_yticklabels(
            [f"{lab} (n={int(row_sums[i, 0])})" for i, lab in enumerate(labels)],
            fontsize=6,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            "Confusion matrix (row-normalised; n = images per true class)"
        )
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class"
        )

        # Annotate raw counts only when there are few enough classes to read.
        if len(labels) <= settings.TRAIN_CM_ANNOTATE_MAX:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    count = int(cm[i, j])
                    if count == 0:
                        continue
                    ax.text(
                        j, i, str(count), ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                    )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Remaining pages: example classified crops per category.
        if examples:
            _classifier_example_pages(pdf, labels, examples)
    logger.info("Wrote classifier stats PDF: %s", pdf_path)


def camsync_summary_figure(summary: dict):
    """A matplotlib Figure with a text summary of the ``substrata camsync`` run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines: list[str] = []
    s = summary

    def add(label: str, value: object) -> None:
        lines.append(f"{label}: {value}")

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle("Camsync - run summary", fontsize=12, y=0.98)

    def _fmt_cli(val: object) -> str:
        return "(not set)" if val is None else repr(val)

    add("Project path", s.get("project_path", ""))
    add("Working directory", s.get("cwd", ""))
    add("Project id", s.get("project_id", ""))
    lines.append("")
    add("Pose source sensor_id", s.get("pose_source", ""))
    add("Updated target sensor_id", s.get("updated_target", ""))
    lines.append("")
    ox = s.get("offset_xyz")
    if ox is not None:
        oa = np.asarray(ox, dtype=float).ravel()
        add(
            "offset_xyz (pose-local m; applied to sync)",
            f"[{oa[0]:.6g}, {oa[1]:.6g}, {oa[2]:.6g}]",
        )
    else:
        add("offset_xyz (pose-local)", "none (poses copied without xyz offset)")
    lines.append("")
    add("time_offset_sec (applied to target EXIF)", s.get("time_offset_sec", ""))
    add("scale_factor (project / initializer)", s.get("scale_factor", ""))
    lines.append("")
    add("Flag --auto-offsets", s.get("auto_offsets", ""))
    add("Flag --auto-time", s.get("auto_time", ""))
    add("Flag --auto-xyz", s.get("auto_xyz", ""))
    add("Flag --yes", s.get("assume_yes", ""))
    add("Flag --local", s.get("local", ""))
    lines.append("")
    add("spatial_max_dist (m, for --auto-time)", s.get("spatial_max_dist", ""))
    add("min_spatial_pairs (for --auto-time)", s.get("min_spatial_pairs", ""))
    lines.append("")
    add(
        "CLI --time-offset (ignored if --auto-time)",
        _fmt_cli(s.get("cli_time_offset")),
    )
    add("CLI --xyz (ignored if --auto-xyz)", _fmt_cli(s.get("cli_xyz")))
    lines.append("")
    add("PDF intercept_search_radius", s.get("intercept_search_radius", ""))
    add("Point cloud loaded (intercept highlights)", s.get("pcd_loaded", ""))

    text = "\n".join(lines)
    ax.text(
        0.06,
        0.94,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )
    return fig


def write_camsync_sanity_pdf(
    pose_cams,
    updated_cams,
    output_path: str,
    *,
    time_offset_sec: float,
    pcd=None,
    camsync_summary: dict | None = None,
    max_pairs: int = settings.CAMSYNC_PDF_PAIRS,
    intercept_search_radius: float = 0.01,
) -> None:
    """Write a PDF: first a text summary page, then pose/target image pairs.

    Pairs are spread over pages of at most ``settings.CAMSYNC_PDF_PAIRS_PER_PAGE``
    rows each.

    Titles use raw EXIF from each file (``get_datetime_original(None)``). Brackets
    label ``dt`` (seconds) applied for sync on the target side only (pose uses 0 s).
    If
    ``pcd`` is set, draws the same PCD intercept (ray from target camera along its
    viewing axis) highlighted on both images, matching ``sandbox_macros`` notebook
    logic.

    Args:
        pose_cams: Pose-source :class:`~substrata.cameras.Cameras` subset.
        updated_cams: Updated-target :class:`~substrata.cameras.Cameras` subset.
        output_path: Destination ``.pdf`` path.
        time_offset_sec: Seconds applied to updated-target EXIF for time alignment.
        pcd: Optional loaded :class:`~substrata.pointclouds.PointCloud` for intercepts.
        camsync_summary: Optional dict for the first-page run summary (see
            :func:`camsync_summary_figure`).
        max_pairs: Number of match rows to render (default
            ``settings.CAMSYNC_PDF_PAIRS``).
        intercept_search_radius: Step radius for :meth:`PointCloud.get_intercept`
            (default 0.01, same as sandbox notebook).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image, ImageDraw

    if pcd is None:
        print(
            "Camsync sanity PDF: no point cloud loaded; "
            "intercept highlights skipped."
        )

    targets = sorted(
        updated_cams.data.values(),
        key=lambda c: (c.datetime is None, str(c.datetime), str(c.cam_id)),
    )
    pairs: list[tuple] = []
    for tcam in targets:
        pcam = pose_cams.get_camera_by_datetime(tcam.datetime)
        if pcam is not None:
            pairs.append((pcam, tcam))
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        print("Camsync sanity PDF skipped: no time-matched pairs with pose cameras.")
        return

    def _exif_title_line(cam, is_target: bool) -> str:
        raw = cam.get_datetime_original(None)
        if raw is None:
            raw = "(no EXIF DateTimeOriginal)"
        k = float(time_offset_sec)
        if is_target:
            bracket = f"(sync dt={k:+.4g} s to target EXIF)"
        else:
            bracket = "(sync dt=0 s)"
        return f"{raw} {bracket}"

    def _pixel_highlight_from_cam(cam, world_pt: np.ndarray):
        px = cam.get_pixel_coords(
            world_pt, required_to_be_in_view=False, use_orig_coords=False
        )
        if px is None or px[0] is None:
            return None
        return (int(px[0]), int(px[1]))

    def _rgb_with_highlight(fp: str, highlight_xy: tuple[int, int] | None):
        if not fp or not os.path.isfile(fp):
            return None
        try:
            image = Image.open(fp).convert("RGB")
        except OSError:
            return None
        if highlight_xy is not None:
            x, y = highlight_xy
            w, h = image.size
            r = max(12, min(50, min(w, h) // 8))
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 0, 0),
            )
        return np.asarray(image)

    pdf = PdfPages(output_path)
    if camsync_summary is not None:
        fig_sum = camsync_summary_figure(camsync_summary)
        pdf.savefig(fig_sum)
        plt.close(fig_sum)

    n_total = len(pairs)
    per_page = max(1, int(settings.CAMSYNC_PDF_PAIRS_PER_PAGE))
    for start in range(0, n_total, per_page):
        page_pairs = pairs[start : start + per_page]
        n = len(page_pairs)
        fig, axes = plt.subplots(n, 2, figsize=(10, max(2.5, 2.6 * n)))
        axes = np.atleast_2d(axes)

        for i, (pose_cam, tgt_cam) in enumerate(page_pairs):
            pose_hl = None
            tgt_hl = None
            if pcd is not None:
                origin = np.asarray(tgt_cam.coords, dtype=float).ravel()[:3]
                icpt = pcd.get_intercept(
                    origin,
                    intercept_search_radius,
                    vector=tgt_cam.vector,
                )
                if icpt is not None:
                    world_pt = np.asarray(icpt.coords, dtype=float).ravel()[:3]
                    pose_hl = _pixel_highlight_from_cam(pose_cam, world_pt)
                    tgt_hl = _pixel_highlight_from_cam(tgt_cam, world_pt)

            for j, (cam, col_title, is_tgt, hl) in enumerate(
                (
                    (pose_cam, "source (pose)", False, pose_hl),
                    (tgt_cam, "target", True, tgt_hl),
                )
            ):
                ax = axes[i, j]
                fp = cam.filepath
                rgb = _rgb_with_highlight(fp, hl) if fp else None
                if rgb is not None:
                    ax.imshow(rgb)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "(missing or unreadable image)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
                ax.set_axis_off()
                short = os.path.basename(fp) if fp else "(no path)"
                dt_line = _exif_title_line(cam, is_tgt)
                ax.set_title(f"{col_title}\n{dt_line}\n{short}", fontsize=7)

        page_range = f"pairs {start + 1}-{start + n} of {n_total}"
        fig.suptitle(
            "Camsync sanity check: pose vs target pairs (EXIF time match), "
            f"{page_range}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Wrote camsync sanity PDF: {output_path} ({n_total} pairs)")


def save_curation_sheets_pdf(queues, tiles, pdf_path, max_per_sheet):
    """Contact sheets of the curation review queues (``substrata curate --sheets``).

    Each queue's first ``max_per_sheet`` points are drawn from their stored
    tiles with a cross-hair on the point, captioned ``label>predicted p_label``,
    one titled group per queue (:func:`save_training_crops_pdf`).

    Args:
        queues: ``{title: rows}`` of ``curate/points.csv`` rows (dicts with
            ``tile``, ``tile_x``, ``tile_y``, ``id``, ``label``, ``cv_pred``,
            ``p_label``), in review order.
        tiles: The curation tile store (``tiles.get(name) -> (jpeg bytes, ...)``).
        pdf_path: Output PDF path.
        max_per_sheet: Points per queue.

    Returns:
        int: The number of pages written.
    """
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    with tempfile.TemporaryDirectory() as tmp:
        groups = {}
        for title, items in queues.items():
            out = []
            for r in items[:max_per_sheet]:
                dst = os.path.join(tmp, r["tile"])
                tile = tiles.get(r["tile"])
                if tile is None:
                    continue
                if not os.path.isfile(dst):
                    try:
                        with Image.open(BytesIO(tile[0])) as im:
                            im = im.convert("RGB")
                            d = ImageDraw.Draw(im)
                            x, y = f(r["tile_x"]), f(r["tile_y"])
                            if np.isfinite(x) and np.isfinite(y):
                                for c, w in (((0, 0, 0), 3), ((255, 60, 60), 1)):
                                    d.line([(x - 9, y), (x - 3, y)], fill=c, width=w)
                                    d.line([(x + 3, y), (x + 9, y)], fill=c, width=w)
                                    d.line([(x, y - 9), (x, y - 3)], fill=c, width=w)
                                    d.line([(x, y + 3), (x, y + 9)], fill=c, width=w)
                            im.save(dst, quality=90)
                    except OSError:
                        continue
                caption = f"{r['label']}>{r['cv_pred'] or '?'} {f(r['p_label']):.2f}"
                out.append((dst, r["id"], caption, ""))
            groups[title] = (len(items), out)
        return save_training_crops_pdf(groups, pdf_path)
