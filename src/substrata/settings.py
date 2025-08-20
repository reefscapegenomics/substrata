OUTPUT_FOLDER = "output"

DEFAULT_INLIER_RANGE = 0.01
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
