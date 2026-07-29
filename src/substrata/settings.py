import re

OUTPUT_FOLDER = "output"

# measure_all parallelism. Capped (rather than -1 = all cores) because the
# measurements are memory-heavy: gap fraction shares the full point cloud across
# threads, and the plotly/Kaleido QC images (when enabled) each spawn a headless
# browser subprocess. Running one per core exhausts RAM on many-core machines.
DEFAULT_MEASURE_N_JOBS = 4

DEFAULT_INLIER_RANGE = 0.01
DEFAULT_TPI_RADIUS_INNER = 0.1  # metres; excludes focal object from neighbourhood
DEFAULT_TPI_RADIUS_OUTER = 0.5  # metres; outer limit of annulus neighbourhood
CANOPY_COVER_POINT_SPACING = 0.01

# calc_benthic_fraction: annulus neighbourhood + grid sampling of the surface.
DEFAULT_BENTHIC_RADIUS_INNER = 0.1  # metres; excludes the focal object
DEFAULT_BENTHIC_RADIUS_OUTER = 0.5  # metres; outer limit of the sampled annulus
DEFAULT_BENTHIC_SAMPLE_SPACING = 0.05  # metres; XY grid spacing of sample points
DEFAULT_BENTHIC_INTERCEPT_RADIUS = 0.02  # metres; XY search radius for z-intercept
# Height-weighted "interaction cover": colony base level + below-base weight falloff.
DEFAULT_BENTHIC_BASE_PERCENTILE = 10  # percentile of inner-radius z = colony base level
DEFAULT_BENTHIC_BASE_FALLOFF = 0.20  # m; depth below base where height weight reaches 0

# --- OrthoGrid (gridded per-cell rasters: DEM / density / labels) --------------
DEFAULT_ORTHO_CELL_SIZE = 0.1  # metres; default OrthoGrid cell side length
DEFAULT_SEG_CELL_SIZE = 0.05  # metres; default point-cloud segmentation query spacing

# --- Animation (animated matplotlib figures) ----------------------------------
DEFAULT_ANIM_FPS = 15          # frames per second for exported GIF/MP4
DEFAULT_ANIM_DURATION = 5.0    # seconds; default fill length of an animation
DEFAULT_ANIM_SECONDS_PER_CATEGORY = 1.0  # sweep="categories" default per class

ANN_DEFAULT_COL_ORDER = {
    "id": 0,
    "orig_x": 1,
    "orig_y": 2,
    "orig_z": 3,
    "label": 4,
}

FIREFISH_DEFAULT_COLS = [
    "date",
    "time",
    "unixtime",
    "depth",
    "altitude",
    "altitude_conf",
]
CAM_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"

ANN_ID_POST_FIXES = ["_left", "_right"]

LEN_ORIENT_LINE = 10

RANSAC_N = 3
RANSAC_ITERATIONS = 1000

MAX_DIST_FROM_ORIGIN_FOR_INTERCEPT_SEARCH = 5
MAX_SEARCH_RADIUS_FOR_INTERCEPT_SEARCH = 0.1

DEFAULT_INTERCEPT_SEARCH_RADIUS = 0.005
DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN = 0.01
DEFAULT_REPROJECTION_THRESHOLD_DISCARD = 0.5

DEFAULT_PIXEL_REPROJECTION_THRESHOLD = 2

DEFAULT_DEPTH_ACCURACY_THRESHOLD = 0.3
FIREFISH_DEPTH_ALTITUDE_OUTLIER_THRESHOLD = 4.0
FIREFISH_MIN_NUM_CAM_MATCHES = 100

# --- Classifier training (substrata train) ------------------------------------
TRAIN_DEFAULT_PATTERN = "*_slope_intercepts.csv"
TRAIN_CLASSES_FILE = "classes.csv"
TRAIN_IMAGE_SIZE = 224  # pixels; classifier input resolution (fastai item resize)
# Default crop equals the classifier input, so crops are cut at the exact size
# the model consumes (no down-scaling of a larger region).
TRAIN_CROP_SIZE = TRAIN_IMAGE_SIZE  # pixels; square crop centred on (cam_x, cam_y)
TRAIN_CROP_JPEG_QUALITY = 95
TRAIN_CROP_JOBS = -1  # parallel workers for crop generation (-1 = all cores)
TRAIN_DELETE_PREVIEW = 5  # example paths shown before deleting redundant crops
# Tree entries with at least this many occurrences are shown; --min-count only
# controls which shown entries are bolded (i.e. used as training labels).
TRAIN_MIN_VISIBLE_COUNT = 1
TRAIN_SPLIT = (80, 10, 10)  # train / validation / test percentages
TRAIN_CROP_DIRS = ("training_crops", "validation_crops", "test_crops")
TRAIN_DEFAULT_ARCH = "resnet34"
TRAIN_DEFAULT_EPOCHS = 10
TRAIN_DEFAULT_MODEL_FILE = "crop_classifier.pkl"
TRAIN_ANNOTATIONS_FILE = "training_annotations.csv"
# Editable label->training-class map (selection + hierarchical collapse). Seeded
# from the label tree, hand-tunable, and read by both training and evaluation.
TRAIN_LABEL_MAP_FILE = "training_label_map.csv"
TRAIN_SUMMARY_FILE = "training_summary.pdf"  # per-run settings + per-class counts
TRAIN_SUMMARY_CLASSES_PER_PAGE = 40  # category bars per training-summary PDF page
TRAIN_CM_ANNOTATE_MAX = 25  # max classes for per-cell count labels on the matrix
TRAIN_EXAMPLES_PER_CLASS = 10  # example crops shown per category in the PDF
TRAIN_EXAMPLE_ROWS_PER_PAGE = 8  # category rows per example-images PDF page

# Label-tree (CATAMI hierarchy) rendering and consolidated-CSV layout.
TRAIN_LABEL_COLUMN = "label"
TRAIN_LEVEL_COLUMNS = [f"CATAMI_LEVEL_{i}" for i in range(1, 8)]
TRAIN_CROP_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
TRAIN_ANN_COLUMNS = [
    "id",
    "orig_x",
    "orig_y",
    "orig_z",
    "label",
    "label_conf",
    "world_x",
    "world_y",
    "world_z",
    "cam_filepath",
    "cam_x",
    "cam_y",
    "depth",
]

# ANSI styling for bolded (training-label) entries in terminal tree output.
TRAIN_BOLD = "\033[1m"
TRAIN_RESET = "\033[0m"
TRAIN_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

RGL_SCALEBARS = [
    ["target 3", "target 4", 0.500],
    ["target 5", "target 6", 0.500],  # w/ yellow ruler
    ["target 7", "target 8", 0.499],  # w/ color scale
    ["target 31", "target 33", 0.20],  # top-left
    ["target 32", "target 34", 0.20],
    ["target 31", "target 32", 0.07],
    ["target 33", "target 34", 0.07],
    ["target 35", "target 37", 0.20],  # bot-left
    ["target 36", "target 38", 0.20],
    ["target 35", "target 36", 0.07],
    ["target 37", "target 38", 0.07],
    ["target 39", "target 41", 0.20],  # top-right
    ["target 40", "target 42", 0.20],
    ["target 39", "target 40", 0.07],
    ["target 41", "target 42", 0.07],
    ["target 43", "target 45", 0.20],  # bot-right
    ["target 44", "target 46", 0.20],
    ["target 43", "target 44", 0.07],
    ["target 45", "target 46", 0.07],
]

# --- ColorChecker Classic (24 patches) ----------------------------------------
# Reference sRGB (D65, 8-bit) from published ColorChecker tables; see e.g.
# https://www.babelcolor.com/colorchecker-2.htm (BabelColor / X-Rite references).
# Patch (u, v) are normalized chart coordinates: u along top edge (6 columns),
# v down the left edge (4 rows), both in [0, 1].
#
# Rows are top-to-bottom; columns left-to-right (X-Rite layout).
COLORCHECKER_CLASSIC_PATCHES = [
    # row 0
    ("dark skin", 1 / 12, 1 / 8, 115, 82, 68),
    ("light skin", 3 / 12, 1 / 8, 194, 150, 130),
    ("blue sky", 5 / 12, 1 / 8, 98, 122, 157),
    ("foliage", 7 / 12, 1 / 8, 87, 108, 67),
    ("blue flower", 9 / 12, 1 / 8, 133, 128, 177),
    ("bluish green", 11 / 12, 1 / 8, 103, 189, 170),
    # row 1
    ("orange", 1 / 12, 3 / 8, 214, 126, 44),
    ("purplish blue", 3 / 12, 3 / 8, 80, 91, 166),
    ("moderate red", 5 / 12, 3 / 8, 193, 90, 99),
    ("purple", 7 / 12, 3 / 8, 94, 60, 108),
    ("yellow green", 9 / 12, 3 / 8, 157, 188, 64),
    ("orange yellow", 11 / 12, 3 / 8, 224, 163, 46),
    # row 2
    ("blue", 1 / 12, 5 / 8, 56, 61, 150),
    ("green", 3 / 12, 5 / 8, 70, 148, 73),
    ("red", 5 / 12, 5 / 8, 175, 54, 60),
    ("yellow", 7 / 12, 5 / 8, 231, 199, 31),
    ("magenta", 9 / 12, 5 / 8, 187, 86, 149),
    ("cyan", 11 / 12, 5 / 8, 8, 133, 161),
    # row 3
    ("white", 1 / 12, 7 / 8, 243, 243, 242),
    ("neutral 8", 3 / 12, 7 / 8, 200, 200, 200),
    ("neutral 6.5", 5 / 12, 7 / 8, 160, 160, 160),
    ("neutral 5", 7 / 12, 7 / 8, 122, 122, 121),
    ("neutral 3.5", 9 / 12, 7 / 8, 85, 85, 85),
    ("black", 11 / 12, 7 / 8, 52, 52, 52),
]

# Marker-quad bounds in patch-grid UV space.
#
# Patch UV [0,1]×[0,1] maps to the printed 6×4 colour-patch grid, which is
# SMALLER than the full card (there is a border/frame around the patches).
# These four values locate the TL and BR survey targets in that same UV space.
# Values outside [0,1] mean the target sits outside the printed patches.
#
# All distances below are in mm; fine-tune from the plane-view plot.
#   Full card:          130 × 75 mm
#   Printed patch grid: ~102 × ~50 mm
#   Marker quad span:   200 (u) × 70 (v) mm  (from scalebar pairs)
#   TL target → left patch edge:  ~49 mm
#   TL target → top patch edge:   ~15 mm
_PATCH_GRID_W = 90
_PATCH_GRID_H = 60
_TL_TO_PATCHES_LEFT = 52
_TL_TO_PATCHES_TOP = 5
_MARKER_SPAN_U = 200
_MARKER_SPAN_V = 70

COLORCHECKER_MARKER_U_MIN = -_TL_TO_PATCHES_LEFT / _PATCH_GRID_W
COLORCHECKER_MARKER_V_MIN = -_TL_TO_PATCHES_TOP / _PATCH_GRID_H
COLORCHECKER_MARKER_U_MAX = (_MARKER_SPAN_U - _TL_TO_PATCHES_LEFT) / _PATCH_GRID_W
COLORCHECKER_MARKER_V_MAX = (_MARKER_SPAN_V - _TL_TO_PATCHES_TOP) / _PATCH_GRID_H

# Sampling and QC (units match the point cloud).
DEFAULT_COLOR_CALIBRATION_RADIUS = 0.005
DEFAULT_COLOR_CALIBRATION_PLANE_EPSILON = 0.003
COLOR_CALIBRATION_OUTLIER_Z = 2.5
# Minimum number of points (after radius + plane filtering) required for a
# patch to produce a measured median; below this, the patch is treated as
# "no data" and excluded from aggregation/QC.  Prevents 1-2 spurious points
# captured outside the chart from producing a garbage measurement.
DEFAULT_COLOR_CALIBRATION_MIN_POINTS = 5

# Per card: [tl, bl, tr, br, name]  (name is optional).
RGL_COLOR_CALIBRATIONS = [
    ["target 34", "target 33", "target 32", "target 31", "top-left"],
    ["target 38", "target 37", "target 36", "target 35", "bottom-left"],
    ["target 42", "target 41", "target 40", "target 39", "top-right"],
    ["target 43", "target 44", "target 45", "target 46", "bottom-right"],
]
