"""Stratified PLY: a point cloud's PLY, reordered for progressive and regional reads.

A stratified PLY is still the project's ordinary binary PLY (``<id>.ply``, same
header and byte-identical records), only reordered, with a small sidecar index
``<id>.strat.npz``. The points are split into spatially compact clusters (the
strata, from a kd-split of a count grid), each cluster is shuffled, and the file
holds round 0 of every cluster, then round 1 of every cluster, and so on; round r
holds a fraction 2^(r-R+1) (cumulative) of each cluster. Consequences:

* the first N records are a uniform random subsample of the whole cloud, so a
  decimated cloud is a prefix read (no separate ``_dec50M.ply``);
* any cluster at detail level r is r contiguous slices, so a region can be read
  at any density without scanning the file (``StratifiedPly.query_box``);
* every record is byte-identical to the input, so other PLY readers still work.

A PLY counts as stratified when its ``.strat.npz`` exists and matches the file
(``is_stratified``). Build one with ``stratify_ply`` (``substrata stratify``).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from substrata.logging import logger

__all__ = [
    "PlyHeader",
    "read_ply_header",
    "open_ply_records",
    "ply_xyz",
    "ply_rgb",
    "StratifiedIndex",
    "stratified_index_path",
    "is_stratified",
    "StratifiedPly",
    "stratify_ply",
    "stratify_space",
    "original_backup_path",
]

STRATIFIED_INDEX_SUFFIX = ".strat.npz"
STRATIFIED_INDEX_VERSION = 1
STRATIFIED_COMMENT = b"comment substrata stratified v1\n"
_GRID_CELLS = 1 << 24

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


# ---------------------------------------------------------------------------
# Binary PLY access (vertex-only clouds, as pointclouds._parse_ply_header)
# ---------------------------------------------------------------------------


@dataclass
class PlyHeader:
    """Header of a binary, vertex-only PLY.

    Attributes:
        path: File path.
        fmt: ``binary_little_endian`` or ``binary_big_endian``.
        endian: ``<`` or ``>``.
        n_vertices: Number of vertex records.
        props: Vertex properties as (ply type, name).
        header_len: Header size in bytes (including the end_header line).
        lines: The raw header lines.
    """

    path: str
    fmt: str
    endian: str
    n_vertices: int
    props: List[Tuple[str, str]]
    header_len: int
    lines: List[bytes] = field(repr=False)

    @property
    def dtype(self) -> np.dtype:
        """Structured dtype of one vertex record."""
        return np.dtype(
            [(name, self.endian + _PLY_TYPES[t]) for t, name in self.props]
        )

    @property
    def record_size(self) -> int:
        """Bytes per vertex record."""
        return self.dtype.itemsize

    def has(self, *names: str) -> bool:
        """Whether the vertices have all these properties."""
        return all(n in self.dtype.names for n in names)


def read_ply_header(path: str) -> PlyHeader:
    """Parse the header of a binary, vertex-only PLY.

    Args:
        path: PLY file path.

    Returns:
        The parsed header.

    Raises:
        ValueError: For ASCII PLYs, non-empty non-vertex elements, list
            properties on vertices or a malformed header.
    """
    lines: List[bytes] = []
    fmt = endian = None
    n_vertices = None
    props: List[Tuple[str, str]] = []
    in_vertex = False
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        lines.append(b"ply\n")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            lines.append(line)
            s = line.decode("ascii").strip()
            if s.startswith("format "):
                fmt = s.split()[1]
                if fmt == "binary_little_endian":
                    endian = "<"
                elif fmt == "binary_big_endian":
                    endian = ">"
                else:
                    raise ValueError(f"{path}: only binary PLY is supported")
            elif s.startswith("element "):
                _, name, count = s.split()[:3]
                in_vertex = name == "vertex"
                if in_vertex:
                    n_vertices = int(count)
                elif int(count) != 0:
                    raise ValueError(f"{path}: non-empty element '{name}' unsupported")
            elif s.startswith("property ") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices unsupported")
                props.append((parts[1], parts[2]))
            elif s == "end_header":
                break
        header_len = f.tell()
    if fmt is None or n_vertices is None or not props:
        raise ValueError(f"{path}: invalid PLY header")
    return PlyHeader(path, fmt, endian, n_vertices, props, header_len, lines)


def open_ply_records(
    header: PlyHeader, mode: str = "r", path: Optional[str] = None
) -> np.memmap:
    """Memory-map the vertex records of a PLY.

    Args:
        header: Its header (``read_ply_header``).
        mode: numpy memmap mode.
        path: Another file with the same layout (default: ``header.path``).
    """
    return np.memmap(
        path or header.path,
        dtype=header.dtype,
        mode=mode,
        offset=header.header_len,
        shape=(header.n_vertices,),
    )


def ply_xyz(records: np.ndarray) -> np.ndarray:
    """Return (N, 3) float64 positions of structured vertex records."""
    out = np.empty((len(records), 3), np.float64)
    out[:, 0] = records["x"]
    out[:, 1] = records["y"]
    out[:, 2] = records["z"]
    return out


def ply_rgb(records: np.ndarray) -> np.ndarray:
    """Return (N, 3) uint8 colours of structured vertex records.

    Float colours in [0, 1] are rescaled, wider integers shifted down; grey
    when the records have no colour.
    """
    names = records.dtype.names
    if not all(c in names for c in ("red", "green", "blue")):
        return np.full((len(records), 3), 180, np.uint8)
    out = np.empty((len(records), 3), np.uint8)
    for i, c in enumerate(("red", "green", "blue")):
        v = records[c]
        if v.dtype.kind == "f":
            out[:, i] = np.clip(v * 255.0 + 0.5, 0, 255)
        elif v.dtype.itemsize > 1:
            out[:, i] = np.clip(v >> (8 * (v.dtype.itemsize - 1)), 0, 255)
        else:
            out[:, i] = v
    return out


def _stratified_header_bytes(header: PlyHeader) -> bytes:
    """The header with the stratified marker comment after the format line."""
    out = []
    for line in header.lines:
        if line == STRATIFIED_COMMENT:
            continue
        out.append(line)
        if line.startswith(b"format "):
            out.append(STRATIFIED_COMMENT)
    return b"".join(out)


# ---------------------------------------------------------------------------
# Sidecar index
# ---------------------------------------------------------------------------


def stratified_index_path(ply_path: str) -> str:
    """Return the sidecar index path of a PLY (``<root>.strat.npz``)."""
    return os.path.splitext(ply_path)[0] + STRATIFIED_INDEX_SUFFIX


@dataclass
class StratifiedIndex:
    """Where every cluster's rounds are in a stratified PLY.

    Attributes:
        n: Number of records.
        header_len: PLY header size in bytes.
        record_size: Bytes per record.
        rounds: Number of rounds (R).
        counts: (R, C) records of cluster c in round r.
        starts: (R, C) record offset of that slice.
        bbox_lo: (C, 3) cluster bounding box minimum.
        bbox_hi: (C, 3) cluster bounding box maximum.
        area: (C,) approximate surface area per cluster.
        file_size: Size of the PLY when the index was written.
        file_mtime_ns: Its modification time then.
    """

    n: int
    header_len: int
    record_size: int
    rounds: int
    counts: np.ndarray
    starts: np.ndarray
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    area: np.ndarray
    file_size: int
    file_mtime_ns: int

    @property
    def n_clusters(self) -> int:
        """Number of clusters (C)."""
        return self.counts.shape[1]

    @property
    def cluster_sizes(self) -> np.ndarray:
        """(C,) records per cluster."""
        return self.counts.sum(0)

    @property
    def round_ends(self) -> np.ndarray:
        """(R,) record index where each round ends (prefix covering rounds <= r)."""
        return np.cumsum(self.counts.sum(1))

    def save(self, path: str) -> None:
        """Write the index as an .npz file."""
        np.savez(path, version=STRATIFIED_INDEX_VERSION, **asdict(self))

    @classmethod
    def load(cls, path: str) -> "StratifiedIndex":
        """Read an index written by ``save``."""
        z = np.load(path)
        if int(z["version"]) != STRATIFIED_INDEX_VERSION:
            raise ValueError(f"{path}: unsupported index version {int(z['version'])}")
        kw = {k: z[k] for k in cls.__dataclass_fields__}
        for k in ("n", "header_len", "record_size", "rounds", "file_size",
                  "file_mtime_ns"):
            kw[k] = int(kw[k])
        return cls(**kw)

    def matches(self, header: PlyHeader, file_size: int) -> bool:
        """Whether this index describes a PLY with this header and size."""
        return (
            file_size == self.file_size
            and header.n_vertices == self.n
            and header.header_len == self.header_len
            and header.record_size == self.record_size
        )


def is_stratified(ply_path: Optional[str]) -> bool:
    """Whether a PLY is stratified: its ``.strat.npz`` exists and matches it.

    Args:
        ply_path: PLY file path (None gives False).
    """
    if not ply_path or not os.path.isfile(ply_path):
        return False
    idx_path = stratified_index_path(ply_path)
    if not os.path.isfile(idx_path):
        return False
    try:
        idx = StratifiedIndex.load(idx_path)
        return idx.matches(read_ply_header(ply_path), os.path.getsize(ply_path))
    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class StratifiedPly:
    """Random access to a stratified PLY through its sidecar index.

    Args:
        path: The PLY (its ``.strat.npz`` must exist and match).

    Raises:
        ValueError: If the PLY is not stratified (or its index is stale).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.header = read_ply_header(path)
        idx_path = stratified_index_path(path)
        if not os.path.isfile(idx_path):
            raise ValueError(
                f"{path}: not stratified (no {os.path.basename(idx_path)}); "
                "run `substrata stratify`"
            )
        self.index = StratifiedIndex.load(idx_path)
        st = os.stat(path)
        if not self.index.matches(self.header, st.st_size):
            raise ValueError(
                f"{path}: {os.path.basename(idx_path)} does not match the file "
                "(re-run `substrata stratify --force`)"
            )
        if abs(st.st_mtime_ns - self.index.file_mtime_ns) > 2_000_000_000:
            # FAT/exFAT round mtimes; copies get new ones
            logger.debug(f"{path}: modified since its index was written")
        self.records = open_ply_records(self.header)

    @property
    def n(self) -> int:
        """Number of points."""
        return self.index.n

    @property
    def dtype(self) -> np.dtype:
        """Structured dtype of the records."""
        return self.header.dtype

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lo, hi) bounding box of the cloud."""
        return self.index.bbox_lo.min(0), self.index.bbox_hi.max(0)

    def summary(self) -> str:
        """A short description: points, properties, bounding box, clusters and rounds.

        The prefix ends are the record counts at which each round is complete,
        i.e. the sizes of the uniform subsamples a prefix read gives.
        """
        idx, (lo, hi) = self.index, self.bbox
        sizes = idx.cluster_sizes
        ends = ", ".join(f"{int(e):,}" for e in idx.round_ends)
        index_name = os.path.basename(stratified_index_path(self.path))
        smin, smed, smax = int(sizes.min()), int(np.median(sizes)), int(sizes.max())
        return "\n".join([
            f"{self.path}  (stratified, {index_name})",
            f"  points   {self.n:,} x {idx.record_size} B  props={self.dtype.names}",
            f"  bbox     {lo.round(3).tolist()} .. {hi.round(3).tolist()}",
            f"  clusters {idx.n_clusters:,}  (points min/median/max "
            f"{smin:,}/{smed:,}/{smax:,})",
            f"  rounds   {idx.rounds}  prefix ends: {ends}",
        ])

    @staticmethod
    def xyz(records: np.ndarray) -> np.ndarray:
        """(N, 3) float64 positions of records (``ply_xyz``)."""
        return ply_xyz(records)

    @staticmethod
    def rgb(records: np.ndarray) -> np.ndarray:
        """(N, 3) uint8 colours of records (``ply_rgb``)."""
        return ply_rgb(records)

    def read_prefix(self, n: int) -> np.ndarray:
        """First n records: a uniform random subsample of the cloud."""
        return np.array(self.records[:n])

    def read_slices(
        self, c: int, r_from: int = 0, r_to: Optional[int] = None
    ) -> np.ndarray:
        """Records of cluster c in rounds [r_from, r_to)."""
        idx = self.index
        r_to = idx.rounds if r_to is None else r_to
        parts = [
            self.records[idx.starts[r, c]:idx.starts[r, c] + idx.counts[r, c]]
            for r in range(r_from, r_to)
        ]
        if not parts:
            return np.empty(0, self.dtype)
        return np.concatenate(parts)

    def read_cluster(self, c: int, rounds: Optional[int] = None) -> np.ndarray:
        """Cluster c with its first `rounds` rounds (default all: full detail)."""
        return self.read_slices(c, 0, rounds)

    def clusters_in_box(self, lo, hi) -> np.ndarray:
        """Indices of the clusters whose bounding box overlaps [lo, hi]."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        inside = (self.index.bbox_hi >= lo) & (self.index.bbox_lo <= hi)
        return np.flatnonzero(inside.all(1))

    def query_box(self, lo, hi, rounds: Optional[int] = None) -> np.ndarray:
        """All records inside the axis-aligned box [lo, hi].

        Only the overlapping clusters are read.

        Args:
            lo: (3,) box minimum.
            hi: (3,) box maximum.
            rounds: Detail level (default all rounds: full resolution).
        """
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out = []
        for c in self.clusters_in_box(lo, hi):
            rec = self.read_cluster(int(c), rounds)
            xyz = ply_xyz(rec)
            out.append(rec[((xyz >= lo) & (xyz <= hi)).all(1)])
        return np.concatenate(out) if out else np.empty(0, self.dtype)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class _Grid:
    origin: np.ndarray  # (3,) float64
    cell: float
    dims: np.ndarray  # (3,) int64

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.dims))

    def cell_ids(self, xyz: np.ndarray) -> np.ndarray:
        ijk = np.floor((xyz - self.origin) / self.cell).astype(np.int64)
        np.clip(ijk, 0, self.dims - 1, out=ijk)
        return (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]


def _make_grid(lo: np.ndarray, hi: np.ndarray, target: int = _GRID_CELLS) -> _Grid:
    ext = np.maximum(hi - lo, 1e-9)

    def n_for(s):
        return np.prod(np.maximum(1, np.ceil(ext / s)))

    a, b = ext.max() / target, ext.max()  # n_for(a) >= target >= n_for(b)
    for _ in range(60):
        m = np.sqrt(a * b)
        a, b = (m, b) if n_for(m) > target else (a, m)
    s = b
    dims = np.maximum(1, np.ceil(ext / s)).astype(np.int64)
    return _Grid(lo.astype(np.float64), float(s), dims)


def _sample_bbox(rec: np.memmap, n_samples: int = 64, per: int = 262_144):
    n = len(rec)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in np.linspace(0, max(0, n - per), n_samples).astype(np.int64):
        xyz = ply_xyz(np.asarray(rec[s:s + per]))
        xyz = xyz[np.isfinite(xyz).all(1)]
        if len(xyz):
            lo = np.minimum(lo, xyz.min(0))
            hi = np.maximum(hi, xyz.max(0))
    pad = (hi - lo) * 0.01 + 1e-6
    return lo - pad, hi + pad


def _kd_clusters(counts: np.ndarray, target: int) -> list:
    """Split a 3D count grid into boxes of <= target points (or single cells).

    Returns the (lo, hi) cell ranges in depth-first, spatially coherent order.
    """
    leaves = []
    stack = [((0, 0, 0), counts.shape)]
    while stack:
        lo, hi = stack.pop()
        sub = counts[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        total = int(sub.sum())
        if total == 0:
            continue
        size = np.subtract(hi, lo)
        if total <= target or (size == 1).all():
            leaves.append((lo, hi))
            continue
        axis = int(np.argmax(size))
        other = tuple(i for i in range(3) if i != axis)
        cs = np.cumsum(sub.sum(axis=other))
        k = int(np.searchsorted(cs, total / 2)) + 1
        k = min(max(k, 1), size[axis] - 1)
        mid_hi = list(hi)
        mid_hi[axis] = lo[axis] + k
        mid_lo = list(lo)
        mid_lo[axis] = lo[axis] + k
        # push right first so left is processed first
        stack.append((tuple(mid_lo), tuple(hi)))
        stack.append((tuple(lo), tuple(mid_hi)))
    return leaves


def _surface_area(xyz: np.ndarray) -> float:
    """Approximate surface area from voxel occupancy at extent/256."""
    lo = xyz.min(0)
    ext = xyz.max(0) - lo
    v = max(float(ext.max()) / 256.0, 1e-9)
    ijk = np.floor((xyz - lo) / v).astype(np.int64)
    keys = (ijk[:, 0] * 257 + ijk[:, 1]) * 257 + ijk[:, 2]
    return float(len(np.unique(keys))) * v * v


def _round_counts(n: np.ndarray, R: int) -> np.ndarray:
    """(R, C) per-round counts; cumulative round r holds ceil(n * 2^(r-R+1))."""
    frac = 2.0 ** (np.arange(R) - (R - 1))
    cum = np.ceil(n[None, :] * frac[:, None]).astype(np.int64)
    cum[-1] = n
    return np.diff(cum, axis=0, prepend=0)


def original_backup_path(ply_path: str) -> str:
    """Where ``stratify_ply(keep_original=True)`` keeps the unstratified PLY:
    ``backup_<name>.ply`` in the same folder."""
    d, name = os.path.split(os.path.abspath(ply_path))
    return os.path.join(d, f"backup_{name}")


def stratify_space(src: str, tmp_dir: Optional[str] = None) -> dict:
    """Disk space that stratifying `src` in place needs, and what is free.

    The cluster-sorted temp file and the new PLY are each as large as the
    source; both are written before the source is replaced (or renamed to
    its backup), so ~2x its size is needed, the temp part on `tmp_dir` if
    given.

    Args:
        src: The PLY.
        tmp_dir: Folder for the temp file (default: next to the PLY).

    Returns:
        {"size", "need", "free", "ok"} in bytes (need/free/ok on the PLY's
        disk; with another tmp_dir also "tmp_need", "tmp_free").
    """
    import shutil

    size = os.path.getsize(src)
    d = os.path.dirname(os.path.abspath(src))
    free = shutil.disk_usage(d).free
    same = tmp_dir is None or os.stat(tmp_dir).st_dev == os.stat(d).st_dev
    out = {"size": size, "need": 2 * size if same else size, "free": free}
    out["ok"] = out["free"] >= out["need"]
    if not same:
        out["tmp_need"], out["tmp_free"] = size, shutil.disk_usage(tmp_dir).free
        out["ok"] = out["ok"] and out["tmp_free"] >= size
    return out


def stratify_ply(
    src: str,
    dst: Optional[str] = None,
    *,
    cluster_size: int = 2_000_000,
    rounds: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    chunk: int = 8_000_000,
    seed: int = 0,
    progress: bool = True,
    keep_original: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Stratify a binary PLY: reorder its records and write ``<root>.strat.npz``.

    Three streaming passes (count, scatter by cluster, shuffle and write rounds),
    so clouds far larger than memory work. Records are checksummed (xyz sums
    and colour sums) before the result replaces anything.

    Args:
        src: Binary, vertex-only PLY.
        dst: Output PLY. Default: ``src`` itself (reordered in place; the result
            is written next to it first, then swapped in). Name it ``<id>.ply``
            to keep the project convention, e.g. a copy on a fast disk.
        cluster_size: Target points per cluster.
        rounds: Number of rounds (default: from the median cluster size, so the
            coarsest round has ~256 points per cluster).
        tmp_dir: Folder for the cluster-sorted temp file (as big as the cloud;
            default: next to the output).
        chunk: Records per streaming chunk.
        seed: Shuffle seed.
        progress: Show progress bars (tqdm, in points).
        keep_original: In place only: keep the unstratified PLY as
            ``backup_<name>.ply`` next to it (a rename, no copy) instead of
            replacing it.
        on_progress: Called with (fraction done 0..1, stage) as the passes run,
            e.g. for a progress display elsewhere.

    Returns:
        The stratified PLY's path.

    Raises:
        RuntimeError: If the checksums of input and output differ.
    """
    t0 = time.time()
    h = read_ply_header(src)
    final = os.path.abspath(dst or src)
    tmp_dir = tmp_dir or os.path.dirname(final)
    rec = open_ply_records(h)
    n = h.n_vertices
    rs = h.record_size
    has_rgb = h.has("red", "green", "blue")
    names = [p for _, p in h.props]
    logger.info(f"stratify {src}: {n:,} vertices x {rs} B, props={names}")

    # pass 0: approximate bbox from samples (outliers are clipped into edge cells)
    lo, hi = _sample_bbox(rec)
    grid = _make_grid(lo, hi)
    logger.info(f"stratify: grid dims={grid.dims.tolist()} cell={grid.cell:.4g}")

    # progress: tqdm bars in points, and on_progress over the three passes
    # (weights roughly their share of the time)
    weights = {"count": (0.0, 0.25), "scatter": (0.25, 0.35), "write": (0.6, 0.4)}

    def bar(stage: str, desc: str):
        pb = tqdm(total=n, desc=desc, unit="pt", unit_scale=True, disable=not progress)
        a, w = weights[stage]

        def step(k: int):
            pb.update(k)
            if on_progress:
                on_progress(a + w * min(pb.n / max(n, 1), 1.0), stage)

        return pb, step

    # pass 1: exact cell counts + checksums
    counts = np.zeros(grid.n_cells, np.int64)
    sum_xyz = np.zeros(3)
    sum_rgb = np.zeros(3, np.int64)
    pb, step = bar("count", "pass 1/3 count")
    for s in range(0, n, chunk):
        r = np.asarray(rec[s:s + chunk])
        step(len(r))
        xyz = ply_xyz(r)
        counts += np.bincount(grid.cell_ids(xyz), minlength=grid.n_cells)
        sum_xyz += xyz.sum(0)
        if has_rgb:
            sum_rgb += ply_rgb(r).sum(0, dtype=np.int64)
    pb.close()

    leaves = _kd_clusters(counts.reshape(tuple(grid.dims)), cluster_size)
    C = len(leaves)
    lut = np.full(tuple(grid.dims), -1, np.int32)
    for i, (a, b) in enumerate(leaves):
        lut[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = i
    lut = lut.ravel()
    occupied = counts > 0
    assert (lut[occupied] >= 0).all()
    n_c = np.bincount(lut[occupied], weights=counts[occupied], minlength=C)
    n_c = n_c.astype(np.int64)
    assert n_c.sum() == n
    del counts
    logger.info(
        f"stratify: {C} clusters, size min/median/max = "
        f"{n_c.min():,}/{int(np.median(n_c)):,}/{n_c.max():,}"
    )

    # pass 2: counting-sort records into a cluster-grouped temp file
    tmp_path = os.path.join(tmp_dir, os.path.basename(final) + ".tmp")
    c_start = np.concatenate([[0], np.cumsum(n_c)[:-1]])
    cursor = c_start.copy()
    tmp = np.memmap(tmp_path, dtype=h.dtype, mode="w+", shape=(n,))
    part = final + ".part"
    try:
        pb, step = bar("scatter", "pass 2/3 scatter")
        for s in range(0, n, chunk):
            r = np.asarray(rec[s:s + chunk])
            step(len(r))
            cid = lut[grid.cell_ids(ply_xyz(r))]
            order = np.argsort(cid, kind="stable")
            r = r[order]
            cc = np.bincount(cid, minlength=C)
            pos = 0
            for c in np.flatnonzero(cc):
                k = int(cc[c])
                tmp[cursor[c]:cursor[c] + k] = r[pos:pos + k]
                cursor[c] += k
                pos += k
        pb.close()
        assert (cursor == c_start + n_c).all()
        tmp.flush()

        # pass 3: shuffle each cluster, write its rounds to their final offsets
        if rounds is None:
            med = max(np.median(n_c), 1)
            rounds = int(np.clip(np.floor(np.log2(med / 256)) + 1, 1, 16))
        R = rounds
        rc = _round_counts(n_c, R)
        starts = np.concatenate([[0], np.cumsum(rc.ravel())[:-1]]).reshape(R, C)
        header = _stratified_header_bytes(h)
        with open(part, "wb") as f:
            f.write(header)
            f.truncate(len(header) + n * rs)
        out = np.memmap(part, dtype=h.dtype, mode="r+", offset=len(header),
                        shape=(n,))
        rng = np.random.default_rng(seed)
        bb_lo = np.zeros((C, 3))
        bb_hi = np.zeros((C, 3))
        area = np.zeros(C)
        chk_xyz = np.zeros(3)
        chk_rgb = np.zeros(3, np.int64)
        pb, step = bar("write", "pass 3/3 shuffle + write")
        for c in range(C):
            blk = np.asarray(tmp[c_start[c]:c_start[c] + n_c[c]])
            step(len(blk))
            blk = blk[rng.permutation(len(blk))]
            xyz = ply_xyz(blk)
            bb_lo[c] = xyz.min(0)
            bb_hi[c] = xyz.max(0)
            area[c] = _surface_area(xyz)
            chk_xyz += xyz.sum(0)
            if has_rgb:
                chk_rgb += ply_rgb(blk).sum(0, dtype=np.int64)
            pos = 0
            for r_ in range(R):
                k = int(rc[r_, c])
                out[starts[r_, c]:starts[r_, c] + k] = blk[pos:pos + k]
                pos += k
        pb.close()
        out.flush()
        del out
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    finally:
        del tmp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    xyz_ok = np.allclose(chk_xyz, sum_xyz, rtol=1e-9, atol=1e-6)
    if not xyz_ok or (chk_rgb != sum_rgb).any():
        os.remove(part)
        raise RuntimeError(
            f"checksum mismatch: xyz {chk_xyz} vs {sum_xyz}, "
            f"rgb {chk_rgb} vs {sum_rgb}"
        )

    del rec
    if keep_original and final == os.path.abspath(src):
        backup = original_backup_path(src)
        os.replace(src, backup)  # same folder: a rename
        logger.info(f"stratify: original kept as {backup}")
    os.replace(part, final)
    if on_progress:
        on_progress(1.0, "done")
    st = os.stat(final)
    idx_path = stratified_index_path(final)
    StratifiedIndex(
        n=n,
        header_len=len(header),
        record_size=rs,
        rounds=R,
        counts=rc,
        starts=starts,
        bbox_lo=bb_lo,
        bbox_hi=bb_hi,
        area=area,
        file_size=st.st_size,
        file_mtime_ns=st.st_mtime_ns,
    ).save(idx_path)
    logger.info(
        f"stratify: wrote {final} (+ {os.path.basename(idx_path)}), R={R}, C={C}, "
        f"{time.time() - t0:.0f}s"
    )
    return final
