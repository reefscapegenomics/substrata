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
(``is_stratified``). Build one with ``stratify_ply`` (``substrata stratify``):
in memory when the RAM allows (one sequential read and write), else streaming.
"""

from __future__ import annotations

import os
import threading
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
    "stratify_mode",
    "stratify_memory_need",
    "available_memory",
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
    occupied = np.zeros(257**3, bool)  # a bitmap: much faster than np.unique
    occupied[keys] = True
    return float(np.count_nonzero(occupied)) * v * v


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


def available_memory() -> int:
    """RAM available to a new allocation without swapping (bytes; 0 if unknown),
    within this process's cgroup memory limits (a systemd service, container
    or cluster job) where it has any."""
    avail = 0
    try:
        import psutil

        avail = int(psutil.virtual_memory().available)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
        except OSError:
            pass
    limit = _cgroup_headroom()
    return avail if limit is None else min(avail, limit)


def _cgroup_headroom() -> Optional[int]:
    """Room below the tightest cgroup v2 memory.max of this process (Linux), or None."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = next(line.split(":", 2)[2].strip() for line in f if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    room = None
    path = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
    while path.startswith("/sys/fs/cgroup"):
        try:
            with open(os.path.join(path, "memory.max")) as f:
                mx = f.read().strip()
            if mx != "max":
                with open(os.path.join(path, "memory.current")) as f:
                    cur = int(f.read())
                # page cache counts towards memory.current but can be reclaimed
                with open(os.path.join(path, "memory.stat")) as f:
                    stat = dict(line.split() for line in f)
                free = int(mx) - cur + int(stat.get("inactive_file", 0))
                room = free if room is None else min(room, free)
        except (OSError, ValueError):
            pass
        if path == "/sys/fs/cgroup":
            break
        path = os.path.dirname(path)
    return None if room is None else max(room, 0)


def _index_dtype(n: int) -> np.dtype:
    return np.dtype(np.uint32 if n < 2**32 else np.int64)


def stratify_memory_need(src: str, chunk: int = 8_000_000) -> int:
    """RAM that ``stratify_ply(mode="memory")`` needs for `src` (bytes): all its
    records, a cluster id and an index per point, and the per-chunk work."""
    h = read_ply_header(src)
    per_point = h.record_size + 4 + _index_dtype(h.n_vertices).itemsize
    return h.n_vertices * per_point + chunk * (h.record_size + 64) + (1 << 30)


def stratify_mode(src: str, mode: str = "auto") -> str:
    """The mode ``stratify_ply`` runs in: "memory" (one sequential read and one
    sequential write; needs ``stratify_memory_need`` RAM free) or "stream"
    (three passes through a temp file; little RAM). "auto": memory if it fits."""
    if mode not in ("auto", "memory", "stream"):
        raise ValueError(f"mode must be auto, memory or stream, not {mode!r}")
    if mode != "auto":
        return mode
    # with some room to spare: the page cache and everything else still need memory
    return "memory" if stratify_memory_need(src) <= 0.9 * available_memory() else "stream"


def stratify_space(
    src: str,
    tmp_dir: Optional[str] = None,
    *,
    work_dir: Optional[str] = None,
    mode: str = "auto",
) -> dict:
    """Disk space (and RAM) that stratifying `src` in place needs, and what is free.

    The new PLY is written before the source is replaced (or renamed to its
    backup), so its size is needed next to it. The stream mode also writes a
    cluster-sorted temp file as large as the source (on `tmp_dir` or
    `work_dir` if given). With a `work_dir`, the new PLY is written there
    first and then copied next to the source.

    Args:
        src: The PLY.
        tmp_dir: Folder for the stream mode's temp file.
        work_dir: Folder (e.g. on a fast disk) for the temp file and the new
            PLY before it is moved next to the source.
        mode: As for ``stratify_ply``.

    Returns:
        {"size", "mode", "need", "free", "ok", "mem_need", "mem_free"}: bytes,
        need/free on the PLY's disk; with a work disk elsewhere also
        "work_need", "work_free".
    """
    import shutil

    size = os.path.getsize(src)
    mode = stratify_mode(src, mode)
    d = os.path.dirname(os.path.abspath(src))
    dev = os.stat(d).st_dev
    need: dict = {}
    if mode == "stream":
        t = tmp_dir or work_dir or d
        need[os.stat(t).st_dev] = need.get(os.stat(t).st_dev, 0) + size
    w = work_dir or d
    need[os.stat(w).st_dev] = need.get(os.stat(w).st_dev, 0) + size
    if os.stat(w).st_dev != dev:
        need[dev] = need.get(dev, 0) + size  # the copy back
    out = {"size": size, "mode": mode, "need": need[dev],
           "free": shutil.disk_usage(d).free,
           "mem_need": stratify_memory_need(src), "mem_free": available_memory()}
    out["ok"] = out["free"] >= out["need"]
    for other, n in need.items():
        if other != dev:
            path = tmp_dir if tmp_dir and os.stat(tmp_dir).st_dev == other else w
            out["work_need"], out["work_free"] = n, shutil.disk_usage(path).free
            out["ok"] = out["ok"] and out["work_free"] >= n
    return out


def stratify_ply(
    src: str,
    dst: Optional[str] = None,
    *,
    mode: str = "auto",
    cluster_size: int = 2_000_000,
    rounds: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    chunk: int = 8_000_000,
    buffer_bytes: Optional[int] = None,
    seed: int = 0,
    progress: bool = True,
    keep_original: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Stratify a binary PLY: reorder its records and write ``<root>.strat.npz``.

    Two modes, with the same result:

    * "memory": reads the whole PLY into RAM once (sequentially), orders it
      there and writes the new PLY sequentially. Fastest by far on hard disks;
      needs ``stratify_memory_need`` bytes of RAM (records + 8 B per point).
    * "stream": three passes (count, scatter by cluster into a temp file,
      shuffle and write rounds) with bounded RAM, so clouds far larger than
      memory work. Put the temp file on a fast disk (`work_dir`) if you can.

    Records are checksummed (xyz sums and colour sums) before the result
    replaces anything.

    Args:
        src: Binary, vertex-only PLY.
        dst: Output PLY. Default: ``src`` itself (reordered in place; the result
            is written next to it first, then swapped in). Name it ``<id>.ply``
            to keep the project convention, e.g. a copy on a fast disk.
        mode: "memory", "stream" or "auto" (memory if it fits in the RAM
            available now, see ``stratify_mode``).
        cluster_size: Target points per cluster.
        rounds: Number of rounds (default: from the median cluster size, so the
            coarsest round has ~256 points per cluster).
        tmp_dir: Folder for the stream mode's cluster-sorted temp file (as big
            as the cloud; default: `work_dir`, else next to the output).
        work_dir: Folder to write the new PLY (and the temp file) in first,
            e.g. on an SSD when the PLY is on a hard disk; it is then copied
            next to the output (one sequential write).
        chunk: Records per streaming chunk.
        buffer_bytes: Stream mode: RAM for write buffers (default: a quarter of
            the available RAM, 256 MB to 8 GB). Larger means fewer seeks.
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
    mode = stratify_mode(src, mode)
    work = os.path.abspath(work_dir) if work_dir else os.path.dirname(final)
    part = os.path.join(work, os.path.basename(final) + ".part")
    rec = open_ply_records(h)
    n = h.n_vertices
    logger.info(
        f"stratify {src}: {n:,} vertices x {h.record_size} B, "
        f"props={[p for _, p in h.props]}, mode={mode}"
    )

    # pass 0: approximate bbox from samples (outliers are clipped into edge cells)
    lo, hi = _sample_bbox(rec)
    del rec
    grid = _make_grid(lo, hi)
    logger.info(f"stratify: grid dims={grid.dims.tolist()} cell={grid.cell:.4g}")

    # progress: tqdm bars in points, and on_progress over the stages
    # (weights roughly their share of the time)
    copy_back = work != os.path.dirname(final)
    weights = (
        {"read": (0.0, 0.4), "shuffle": (0.4, 0.15), "write": (0.55, 0.45)}
        if mode == "memory"
        else {"count": (0.0, 0.25), "scatter": (0.25, 0.35), "write": (0.6, 0.4)}
    )
    if copy_back:
        weights = {k: (a * 0.85, w * 0.85) for k, (a, w) in weights.items()}
        weights["copy"] = (0.85, 0.15)

    def bar(stage: str, desc: str):
        pb = tqdm(total=n, desc=desc, unit="pt", unit_scale=True, disable=not progress)
        a, w = weights[stage]

        def step(k: int):
            pb.update(k)
            if on_progress:
                on_progress(a + w * min(pb.n / max(n, 1), 1.0), stage)

        return pb, step

    job = dict(h=h, grid=grid, part=part, cluster_size=cluster_size, rounds=rounds,
               chunk=chunk, seed=seed, bar=bar)
    try:
        if mode == "memory":
            res = _stratify_memory(src, **job)
        else:
            if buffer_bytes is None:
                buffer_bytes = int(np.clip(available_memory() / 4, 1 << 28, 1 << 33))
            tmp_path = os.path.join(tmp_dir or work, os.path.basename(final) + ".tmp")
            res = _stratify_stream(src, tmp_path=tmp_path, buffer_bytes=buffer_bytes, **job)

        xyz_ok = np.allclose(res["chk_xyz"], res["sum_xyz"], rtol=1e-9, atol=1e-6)
        if not xyz_ok or (res["chk_rgb"] != res["sum_rgb"]).any():
            raise RuntimeError(
                f"checksum mismatch: xyz {res['chk_xyz']} vs {res['sum_xyz']}, "
                f"rgb {res['chk_rgb']} vs {res['sum_rgb']}"
            )
        if copy_back:
            near = final + ".part"
            if os.stat(work).st_dev == os.stat(os.path.dirname(final)).st_dev:
                os.replace(part, near)
            else:
                _copy_file(part, near, h.record_size, bar)
                os.remove(part)
            part = near
    except BaseException:
        for p in {part, final + ".part"}:
            if os.path.exists(p):
                os.remove(p)
        raise

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
        header_len=res["header_len"],
        record_size=h.record_size,
        rounds=res["R"],
        counts=res["rc"],
        starts=res["starts"],
        bbox_lo=res["bb_lo"],
        bbox_hi=res["bb_hi"],
        area=res["area"],
        file_size=st.st_size,
        file_mtime_ns=st.st_mtime_ns,
    ).save(idx_path)
    logger.info(
        f"stratify: wrote {final} (+ {os.path.basename(idx_path)}), R={res['R']}, "
        f"C={len(res['area'])}, {mode}, {time.time() - t0:.0f}s"
    )
    return final


# -- the two modes: same clusters, same shuffles, same file ------------------


def _clusters(counts: np.ndarray, grid: _Grid, cluster_size: int, n: int):
    """Cluster lookup per grid cell and points per cluster, from exact counts."""
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
    logger.info(
        f"stratify: {C} clusters, size min/median/max = "
        f"{n_c.min():,}/{int(np.median(n_c)):,}/{n_c.max():,}"
    )
    return lut, n_c


def _layout(n_c: np.ndarray, rounds: Optional[int]):
    """Rounds R, per-(round, cluster) counts and file offsets (in records)."""
    if rounds is None:
        med = max(np.median(n_c), 1)
        rounds = int(np.clip(np.floor(np.log2(med / 256)) + 1, 1, 16))
    rc = _round_counts(n_c, rounds)
    starts = np.concatenate([[0], np.cumsum(rc.ravel())[:-1]]).reshape(rounds, -1)
    return rounds, rc, starts


class _ClusterStats:
    """Per-cluster bbox and area, and the output checksums."""

    def __init__(self, C: int, has_rgb: bool):
        self.bb_lo, self.bb_hi, self.area = np.zeros((C, 3)), np.zeros((C, 3)), np.zeros(C)
        self.chk_xyz, self.chk_rgb = np.zeros(3), np.zeros(3, np.int64)
        self.has_rgb = has_rgb

    def add(self, c: int, blk: np.ndarray) -> None:
        xyz = ply_xyz(blk)
        self.bb_lo[c], self.bb_hi[c] = xyz.min(0), xyz.max(0)
        self.area[c] = _surface_area(xyz)
        self.chk_xyz += xyz.sum(0)
        if self.has_rgb:
            self.chk_rgb += ply_rgb(blk).sum(0, dtype=np.int64)

    def result(self) -> dict:
        return dict(bb_lo=self.bb_lo, bb_hi=self.bb_hi, area=self.area,
                    chk_xyz=self.chk_xyz, chk_rgb=self.chk_rgb)


def _bytes(a: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(a).view(np.uint8))


def _pwrite(fd: int, a: np.ndarray, offset: int) -> None:
    mv = _bytes(a)
    while len(mv):
        k = os.pwrite(fd, mv, offset)
        mv, offset = mv[k:], offset + k


def _read_full(f, view: memoryview, path: str) -> None:
    """Fill `view` from the (unbuffered) file `f`; reads can come back short."""
    got = 0
    while got < len(view):
        k = f.readinto(view[got:])
        if not k:
            raise EOFError(f"{path}: ends early")
        got += k


def _sequential(f) -> None:
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
    except (AttributeError, OSError):  # not on macOS / Windows
        pass


def _drop_cache(fd: int, offset: int, length: int) -> None:
    """Let the OS drop this (clean) part of a file from the page cache: data
    read or written once, so the cache doesn't compete with the RAM in use."""
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


class _Reader:
    """Runs ``load(*span)`` for each span in turn in a background thread, one
    span ahead of ``next()``, which waits for the current one."""

    def __init__(self, load, spans):
        self.load, self.spans, self.i = load, list(spans), 0
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self._start()

    def _run(self, span) -> None:
        try:
            self.load(*span)
        except BaseException as e:
            self.error = e

    def _start(self) -> None:
        if self.i < len(self.spans):
            self.thread = threading.Thread(target=self._run, args=(self.spans[self.i],), daemon=True)
            self.thread.start()

    def next(self) -> None:
        self.thread.join()
        if self.error:
            raise self.error
        self.i += 1
        self._start()


class _Writer:
    """Sequential writes in a background thread (one block in flight), so the
    gathering of the next block overlaps the disk; written data is synced
    every ~1 GB and dropped from the page cache."""

    def __init__(self, f, sync_every: int = 1 << 30):
        self.f, self.sync_every = f, sync_every
        self.done = self.synced = 0
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def _put(self, mv: memoryview) -> None:
        try:
            while len(mv):
                mv = mv[self.f.write(mv):]
                # (unbuffered: a write can be short)
            self.done = self.f.tell()
            if self.done - self.synced >= self.sync_every:
                getattr(os, "fdatasync", os.fsync)(self.f.fileno())
                _drop_cache(self.f.fileno(), self.synced, self.done - self.synced)
                self.synced = self.done
        except BaseException as e:
            self.error = e

    def wait(self) -> None:
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.error:
            raise self.error

    def write(self, a: np.ndarray) -> None:
        self.wait()
        self.thread = threading.Thread(target=self._put, args=(_bytes(a),), daemon=True)
        self.thread.start()


def _copy_file(src: str, dst: str, record_size: int, bar) -> None:
    """Copy `src` to `dst` in large sequential blocks, with a progress bar."""
    pb, step = bar("copy", "copy to the PLY's folder")
    block = 64 << 20
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        _sequential(fi)
        done = 0
        while True:
            b = fi.read(block)
            if not b:
                break
            fo.write(b)
            done += len(b)
            step(done // record_size - pb.n)
    pb.close()


def _stratify_memory(src, *, h, grid, part, cluster_size, rounds, chunk, seed, bar) -> dict:
    n, rs = h.n_vertices, h.record_size
    has_rgb = h.has("red", "green", "blue")

    # read everything (sequentially), cell ids, exact counts, checksums
    records = np.empty(n, h.dtype)
    raw = memoryview(records.view(np.uint8))
    cells = np.empty(n, np.uint32)  # the grid cell, later the cluster, of each point
    counts = np.zeros(grid.n_cells, np.int64)
    sum_xyz, sum_rgb = np.zeros(3), np.zeros(3, np.int64)
    pb, step = bar("read", "read into memory")
    spans = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
    with open(src, "rb", buffering=0) as f:
        _sequential(f)
        f.seek(h.header_len)

        def load(s, e):  # in a thread: the next chunk is read while this one is processed
            _read_full(f, raw[s * rs:e * rs], src)
            _drop_cache(f.fileno(), h.header_len + s * rs, (e - s) * rs)

        ahead = _Reader(load, spans)
        for s, e in spans:
            ahead.next()
            r = records[s:e]
            xyz = ply_xyz(r)
            cells[s:e] = grid.cell_ids(xyz)
            counts += np.bincount(cells[s:e], minlength=grid.n_cells)
            sum_xyz += xyz.sum(0)
            if has_rgb:
                sum_rgb += ply_rgb(r).sum(0, dtype=np.int64)
            step(e - s)
    pb.close()

    lut, n_c = _clusters(counts, grid, cluster_size, n)
    del counts
    C = len(n_c)
    c_start = np.concatenate([[0], np.cumsum(n_c)[:-1]])

    # counting sort by cluster (stable: source order within a cluster), as the
    # stream mode's scatter; then shuffle each cluster with the same generator
    pb, step = bar("shuffle", "order + shuffle")
    idt = _index_dtype(n)
    perm = np.empty(n, idt)
    cursor = c_start.copy()
    for s in range(0, n, chunk):
        cid = lut[cells[s:s + chunk]]
        order = np.argsort(cid, kind="stable")
        cs = cid[order]
        cc = np.bincount(cid, minlength=C)
        first = np.concatenate([[0], np.cumsum(cc)[:-1]])
        perm[cursor[cs] + (np.arange(len(cs)) - first[cs])] = order + s
        cursor += cc
    del cells
    assert (cursor == c_start + n_c).all()
    rng = np.random.default_rng(seed)
    stats = _ClusterStats(C, has_rgb)
    for c in range(C):
        seg = perm[c_start[c]:c_start[c] + n_c[c]]
        seg[:] = seg[rng.permutation(len(seg))]
        stats.add(c, records[seg])
        step(len(seg))
    pb.close()

    # write round 0 of every cluster, then round 1, ...: one sequential write
    R, rc, starts = _layout(n_c, rounds)
    roff = np.cumsum(rc, axis=0) - rc  # where each round starts within its cluster
    header = _stratified_header_bytes(h)
    pb, step = bar("write", "write")
    with open(part, "wb", buffering=0) as f:
        out = _Writer(f)
        out.write(np.frombuffer(header, np.uint8))
        pending, k_pending = [], 0

        def flush():
            idx = np.concatenate(pending)
            out.write(records[idx])
            step(len(idx))
            pending.clear()

        for r_ in range(R):
            for c in range(C):
                k = int(rc[r_, c])
                if k:
                    a = c_start[c] + roff[r_, c]
                    pending.append(perm[a:a + k])
                    k_pending += k
                    if k_pending >= chunk:
                        flush()
                        k_pending = 0
        if pending:
            flush()
        out.wait()
    pb.close()
    return dict(stats.result(), sum_xyz=sum_xyz, sum_rgb=sum_rgb, R=R, rc=rc,
                starts=starts, header_len=len(header))


def _stratify_stream(src, *, h, grid, part, tmp_path, buffer_bytes, cluster_size, rounds,
                     chunk, seed, bar) -> dict:
    n, rs = h.n_vertices, h.record_size
    has_rgb = h.has("red", "green", "blue")
    rec = open_ply_records(h)

    # pass 1: exact cell counts + checksums
    counts = np.zeros(grid.n_cells, np.int64)
    sum_xyz, sum_rgb = np.zeros(3), np.zeros(3, np.int64)
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

    lut, n_c = _clusters(counts, grid, cluster_size, n)
    del counts
    C = len(n_c)
    c_start = np.concatenate([[0], np.cumsum(n_c)[:-1]])

    # pass 2: counting-sort records into a cluster-grouped temp file; buffered
    # per cluster so the disk sees few large writes rather than many small ones
    fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        cursor = c_start.copy()
        bufs: list = [[] for _ in range(C)]
        buffered = 0

        def flush_clusters():
            for c in range(C):
                if bufs[c]:
                    blk = np.concatenate(bufs[c])
                    _pwrite(fd, blk, int(cursor[c]) * rs)
                    cursor[c] += len(blk)
                    bufs[c].clear()

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
                bufs[c].append(r[pos:pos + k])
                pos += k
            buffered += r.nbytes
            if buffered >= buffer_bytes:
                flush_clusters()
                buffered = 0
        flush_clusters()
        pb.close()
        del rec
        assert (cursor == c_start + n_c).all()

        # pass 3: shuffle each cluster, append its rounds to per-round buffers;
        # each round is a contiguous stretch of the output, written in big blocks
        R, rc, starts = _layout(n_c, rounds)
        header = _stratified_header_bytes(h)
        rng = np.random.default_rng(seed)
        stats = _ClusterStats(C, has_rgb)
        with open(part, "wb") as f, open(tmp_path, "rb", buffering=0) as ft:
            f.write(header)
            out = f.fileno()
            rbufs: list = [[] for _ in range(R)]
            rbytes = [0] * R
            rpos = [len(header) + int(starts[r_, 0]) * rs for r_ in range(R)]
            per_round = max(buffer_bytes // R, 1 << 20)

            def flush_round(r_):
                blk = np.concatenate(rbufs[r_])
                _pwrite(out, blk, rpos[r_])
                rpos[r_] += blk.nbytes
                rbufs[r_].clear()
                rbytes[r_] = 0

            pb, step = bar("write", "pass 3/3 shuffle + write")
            for c in range(C):
                blk = np.empty(int(n_c[c]), h.dtype)
                ft.seek(int(c_start[c]) * rs)
                _read_full(ft, _bytes(blk), tmp_path)
                step(len(blk))
                blk = blk[rng.permutation(len(blk))]
                stats.add(c, blk)
                pos = 0
                for r_ in range(R):
                    k = int(rc[r_, c])
                    if k:
                        rbufs[r_].append(blk[pos:pos + k].copy())
                        rbytes[r_] += k * rs
                        if rbytes[r_] >= per_round:
                            flush_round(r_)
                    pos += k
            for r_ in range(R):
                if rbufs[r_]:
                    flush_round(r_)
            pb.close()
    finally:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return dict(stats.result(), sum_xyz=sum_xyz, sum_rgb=sum_rgb, R=R, rc=rc,
                starts=starts, header_len=len(header))
