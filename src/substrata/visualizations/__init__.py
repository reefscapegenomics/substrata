"""Figures and reports: everything substrata draws, renders or writes to a PDF.

Only figures live here. The numbers they show come from the compute modules
(:mod:`~substrata.measurements`, :mod:`~substrata.color_calibration`,
:mod:`~substrata.classification`, ...), whose objects keep thin delegating
methods (e.g. ``Scalebars.save_pdf``). Import this front module
(``from substrata import visualizations``; ``visualizations.tpi_figure(...)``),
not the submodules, which are an internal layout:

- ``clouds``: interactive and static point-cloud views, the composite views
  PDF, before/after comparisons and the xy PCA plot.
- ``colorcards``: colour-checker card and colour-correction QC figures.
- ``photos``: photos with highlighted pixels, SAM2 mask crops, image-match
  overlays and annotated videos.
- ``depth``: depth-regression and camera-residual figures.
- ``terrain``: elevation angle, roughness, TPI and vector-dispersion figures.
- ``gapfraction``: hemispherical gap-fraction images.
- ``benthic``: benthic-fraction maps, their image-match crops and RGB
  statistics.
- ``sampling``: grid cells of the intercept sampling.
- ``reports``: crop and annotation-match review PDFs, per-annotation
  measurement figures and PDFs, scale-bar figures.
- ``runreports``: reports of CLI runs (camsync, imagesync, training,
  classifier stats, curation sheets).
- ``alignment``: point-cloud and camera correspondence QC of ``substrata align``.
- ``_common``: helpers shared between them (plotly 3-D scene axes and colour
  bars, PDF placeholders).
"""

# Third-Party Libraries
# kept in the flat namespace: notebooks use plt after `from substrata import *`
import matplotlib.pyplot as plt

# Local Modules
# the crop helpers live in imagecrops (data path); re-exported for old callers
from substrata.imagecrops import encode_to_png_buffer, get_crop_img
from substrata.visualizations._common import (
    _compact_colorbar, _resolve_scene_axes, _symmetric_ticks,
)
from substrata.visualizations.clouds import (
    plot, plot_2d, plot_2d_ortho, plot_compare, plot_views, plot_xy_pca, save_views_pdf,
    show,
)
from substrata.visualizations.colorcards import (
    color_calibration_figures, plot_color_before_after, plot_color_correction_summary,
    plot_colorchecker_card_summary, save_color_calibration_pdf,
)
from substrata.visualizations.photos import (
    create_annotated_video, draw_image_matches_within_camera, get_crop_img_from_masks,
    _radius_line_endpoint, save_image_matches_within_camera, show_img,
)
from substrata.visualizations.depth import (
    depth_residual_figures, plot_cam_residuals, plot_depth_regression,
    save_depth_residuals_pdf,
)
from substrata.visualizations.terrain import (
    tpi_figure, visualize_elevation_angle, visualize_roughness, visualize_tpi,
    visualize_vector_dispersion,
)
from substrata.visualizations.gapfraction import (
    render_gap_fraction, render_gap_fraction_axes,
)
from substrata.visualizations.benthic import (
    benthic_fraction_figure, visualize_benthic_fraction,
    visualize_benthic_image_matches, visualize_rgb_stats,
)
from substrata.visualizations.sampling import show_grid_cells
from substrata.visualizations.reports import (
    plot_measurements, plot_text, save_annotation_matches_to_pdf,
    save_cropped_image_matches_to_pdf, save_measurement_visualizations_to_pdf,
    save_scalebars_pdf, scalebar_figures, show_measurement_image,
    show_measurement_images, show_scalebars,
)
from substrata.visualizations.alignment import (
    plot_camera_correspondence, plot_pcd_context, plot_pcd_correspondence,
    save_align_qc_pdf,
)
from substrata.visualizations.runreports import (
    camsync_summary_figure, _classifier_example_pages, save_classifier_stats_pdf,
    save_curation_sheets_pdf, save_training_crops_pdf, save_training_summary_pdf,
    write_camsync_sanity_pdf, write_imagesync_report_pdf,
)

# Explicit, so `substrata`'s flat namespace gets these names and not the
# submodule objects (a package's vars() include its submodules).
__all__ = [
    "benthic_fraction_figure", "camsync_summary_figure", "color_calibration_figures",
    "create_annotated_video", "depth_residual_figures",
    "draw_image_matches_within_camera", "encode_to_png_buffer", "get_crop_img",
    "get_crop_img_from_masks", "plot", "plot_2d", "plot_2d_ortho", "plot_cam_residuals",
    "plot_camera_correspondence", "plot_color_before_after",
    "plot_color_correction_summary", "plot_colorchecker_card_summary", "plot_compare",
    "plot_depth_regression", "plot_measurements", "plot_pcd_context",
    "plot_pcd_correspondence", "plot_text", "plot_views", "plot_xy_pca", "plt",
    "render_gap_fraction", "render_gap_fraction_axes", "save_align_qc_pdf",
    "save_annotation_matches_to_pdf", "save_classifier_stats_pdf",
    "save_color_calibration_pdf", "save_cropped_image_matches_to_pdf",
    "save_curation_sheets_pdf", "save_depth_residuals_pdf",
    "save_image_matches_within_camera", "save_measurement_visualizations_to_pdf",
    "save_scalebars_pdf", "save_training_crops_pdf", "save_training_summary_pdf",
    "save_views_pdf", "scalebar_figures", "show", "show_grid_cells", "show_img",
    "show_measurement_image", "show_measurement_images", "show_scalebars", "tpi_figure",
    "visualize_benthic_fraction", "visualize_benthic_image_matches",
    "visualize_elevation_angle", "visualize_rgb_stats", "visualize_roughness",
    "visualize_tpi", "visualize_vector_dispersion", "write_camsync_sanity_pdf",
    "write_imagesync_report_pdf",
]
