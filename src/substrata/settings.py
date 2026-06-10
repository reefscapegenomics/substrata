OUTPUT_FOLDER = "output"

DEFAULT_INLIER_RANGE = 0.01
DEFAULT_TPI_RADIUS_INNER = 0.1  # metres; excludes focal object from neighbourhood
DEFAULT_TPI_RADIUS_OUTER = 0.5  # metres; outer limit of annulus neighbourhood
CANOPY_COVER_POINT_SPACING = 0.01

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
