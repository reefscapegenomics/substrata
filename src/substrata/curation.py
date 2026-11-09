"""Curation of point-classifier training data (``substrata curate``).

Run in a classifier folder (where ``substrata train`` was run), curation adds
three things to the training points of a point-feature classifier:

Decisions. ``curation.jsonl`` records per point whether it is accepted,
relabelled or excluded (with a reason). It is append-only (one JSON object per
line; the latest line of a point wins, ``clear`` withdraws a decision) and keyed
by the photo's file name and the rounded pixel position, so decisions survive
regenerated training sets, other CSV mixes, renumbered ids and moved photo
folders. ``substrata train`` applies them when it collates the CSVs; copying
the file (or ``--curation PATH``) carries them to another classifier folder.

Scores. :func:`score_training_points` ranks the training points by how likely
they need a look: the cross-validated probability of their own class (label
noise), how much of their neighbourhood looks like their class (a point on a
boundary or a small object makes a poor patch label), how consistent their
nearest neighbours' labels are, and how unusual they are. They are written to
``curate/points.csv`` with a thumbnail of each model input window.

Pools. :func:`mine_pool` samples unlabelled points from photos (e.g. AUV frames
of habitats without annotations), clusters their features and scores their
novelty against the training points. Points labelled from a pool (decisions in
the same store) join the next ``substrata train`` with their own camera
statistics (``domain``).

The interactive grid for making decisions is ``substrata curate --gui``
(substrata-explore); everything here works without it.
"""

from __future__ import annotations

# Standard Library
import csv
import datetime
import glob
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Third-Party Libraries
import numpy as np
from PIL import Image

# Local Modules
from substrata import settings
from substrata.logging import logger

__all__ = [
    "point_key", "row_point_key", "CurationStore", "store_for_folder",
    "TileStore", "tile_store",
    "curation_dir", "pool_csvs", "score_training_points", "mine_pool",
    "read_table", "import_decisions", "ring_offsets", "domain_statistics",
]

POINT_COLUMNS = [
    "idx", "id", "key", "label", "cls", "split", "photo", "x", "y", "scale",
    "domain", "ok", "tile", "tile_x", "tile_y", "cv_pred", "cv_conf", "p_label",
    "ring_agree", "ring_sim", "knn_label", "novelty",
]
POOL_EXTRA_COLUMNS = [
    "key", "cluster", "cluster_dist", "pred_label", "pred_conf", "novelty",
    "tile", "tile_x", "tile_y", "scale", "given_label",
]


# ---------------------------------------------------------------------------
# Keys and the decision store
# ---------------------------------------------------------------------------


def point_key(photo: str, x: Any, y: Any) -> str:
    """Key of a point in the curation store: photo file name + rounded pixel."""
    return (f"{os.path.basename(str(photo).strip())}|"
            f"{int(round(float(x)))}|{int(round(float(y)))}")


def row_point_key(row: Dict[str, Any]) -> Optional[str]:
    """:func:`point_key` of an annotation row (None without camera fields)."""
    fp = (row.get("cam_filepath") or "").strip()
    try:
        return point_key(fp, row["cam_x"], row["cam_y"]) if fp else None
    except (KeyError, TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class CurationStore:
    """Curation decisions: an append-only JSON-lines file (plus read-only extras).

    Each line is ``{"key", "action", "label"?, "reason"?, "orig_label"?, "id"?,
    "photo"?, "x"?, "y"?, "time", "by"?}``; ``action`` is one of
    ``settings.CURATION_ACTIONS``. The latest line per key wins; ``clear``
    withdraws the decision. Extra stores are read first, so the folder's own
    store overrides them.
    """

    def __init__(self, path: Optional[str] = None, extra: Sequence[str] = ()):
        self.path = path
        self.extra = [p for p in extra if p]
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.n_bad_lines = 0
        for p in self.extra + ([path] if path else []):
            self._read(p)

    def _read(self, path: str) -> None:
        if not os.path.isfile(path):
            if path in self.extra:
                raise FileNotFoundError(f"Curation store not found: {path}")
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    self._ingest(e)
                except (ValueError, KeyError, TypeError):
                    self.n_bad_lines += 1
        if self.n_bad_lines:
            logger.warning("%s: %d unreadable line(s) skipped", path, self.n_bad_lines)

    def _ingest(self, e: Dict[str, Any]) -> None:
        key = str(e["key"])
        if e["action"] == "clear":
            self._latest.pop(key, None)
        else:
            self._latest[key] = e

    # -- reading ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._latest)

    def __contains__(self, key: str) -> bool:
        return key in self._latest

    def get(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        """The current decision of a key, or None."""
        return self._latest.get(key) if key else None

    def entries(self) -> List[Dict[str, Any]]:
        """All current decisions."""
        return list(self._latest.values())

    def counts(self) -> Counter:
        """Current decisions per action."""
        return Counter(e["action"] for e in self._latest.values())

    def decision(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The decision applying to an annotation row, or None.

        An ``accept`` only holds while the row still has the label it was
        accepted with; relabel and exclude hold whatever the row's label.
        """
        e = self.get(row_point_key(row))
        if e is None:
            return None
        if e["action"] == "accept":
            orig = (e.get("orig_label") or "").strip()
            if orig and orig != (row.get("label") or "").strip():
                return None
        return e

    def apply(
        self, row: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """``(row as curated or None if excluded, action or None)``."""
        e = self.decision(row)
        if e is None:
            return row, None
        if e["action"] == "exclude":
            return None, "exclude"
        if e["action"] == "relabel":
            out = dict(row)
            out["label"] = e["label"]
            if "label_conf" in e:
                out["label_conf"] = e["label_conf"]
            return out, "relabel"
        return row, e["action"]

    # -- writing ---------------------------------------------------------------
    def append(
        self, entries: Iterable[Dict[str, Any]], by: str = "",
    ) -> List[Dict[str, Any]]:
        """Validate, timestamp and append decisions; returns the lines written.

        Raises:
            ValueError: on an unknown action, a relabel without a label, a
                missing key, or a store opened without a writable path.
        """
        if not self.path:
            raise ValueError("This curation store is read-only (no path).")
        lines = []
        for e in entries:
            e = {k: v for k, v in dict(e).items() if v is not None}
            if not e.get("key"):
                raise ValueError("A curation decision needs a key.")
            if e.get("action") not in settings.CURATION_ACTIONS:
                raise ValueError(
                    f"Unknown curation action {e.get('action')!r}; expected one "
                    f"of {', '.join(settings.CURATION_ACTIONS)}.")
            if e["action"] == "relabel":
                e["label"] = str(e.get("label") or "").strip()
                if not e["label"]:
                    raise ValueError("A relabel decision needs a label.")
            e.setdefault("time", _now())
            if by:
                e.setdefault("by", by)
            lines.append(e)
        if not lines:
            return []
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for e in lines:
                    f.write(json.dumps(e, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            for e in lines:
                self._ingest(e)
        return lines

    def export_csv(self, path: str) -> int:
        """Write the current decisions as CSV; returns the number of rows."""
        cols = ["key", "action", "label", "reason", "orig_label", "id", "photo",
                "x", "y", "time", "by"]
        rows = sorted(self.entries(), key=lambda e: e.get("time", ""))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for e in rows:
                w.writerow(e)
        return len(rows)


def store_for_folder(folder: str, extra: Sequence[str] = ()) -> CurationStore:
    """The curation store of a classifier folder (plus read-only extras)."""
    return CurationStore(os.path.join(folder, settings.CURATION_FILE), extra)


def curation_dir(folder: str) -> str:
    return os.path.join(folder, settings.CURATION_DIR)


def pool_csvs(folder: str) -> List[str]:
    """The pools' point CSVs of a classifier folder (``curate --mine``)."""
    return sorted(glob.glob(os.path.join(
        curation_dir(folder), settings.CURATION_POOLS_DIR, "*.csv")))


def import_decisions(
    store: CurationStore, csv_path: str,
    lookup: Optional[Dict[str, Dict[str, Any]]] = None, by: str = "import",
) -> Tuple[int, List[str]]:
    """Append decisions from a CSV (e.g. edited from contact sheets).

    Columns: ``action`` (accept / relabel / exclude / clear), optional
    ``label`` and ``reason``, and the point as ``key``, or
    ``cam_filepath``/``photo`` + ``cam_x``/``x`` + ``cam_y``/``y``, or ``id``
    (looked up in ``lookup``: id -> table row with ``key``).

    Returns:
        ``(n_appended, problems)``.
    """
    lookup = lookup or {}
    entries, problems = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            action = r.get("action", "").lower()
            key = r.get("key")
            ref = None
            if not key:
                photo = r.get("cam_filepath") or r.get("photo")
                x, y = r.get("cam_x") or r.get("x"), r.get("cam_y") or r.get("y")
                if photo and x and y:
                    try:
                        key = point_key(photo, x, y)
                    except ValueError:
                        key = None
                elif r.get("id"):
                    ref = lookup.get(r["id"])
                    key = ref and ref.get("key")
            if not key:
                problems.append(
                    f"line {n}: no point (key, photo + x + y, or a known id)")
                continue
            if action not in settings.CURATION_ACTIONS:
                problems.append(f"line {n}: unknown action {action!r}")
                continue
            if action == "relabel" and not r.get("label"):
                problems.append(f"line {n}: relabel without a label")
                continue
            e = {"key": key, "action": action}
            for k in ("label", "reason", "id"):
                if r.get(k):
                    e[k] = r[k]
            if ref:
                e.setdefault("orig_label", ref.get("label") or "")
                e.setdefault("id", ref.get("id"))
            entries.append(e)
    store.append(entries, by=by)
    return len(entries), problems


def read_table(path: str) -> List[Dict[str, str]]:
    """Rows of a curate table (``points.csv`` or a pool CSV)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tiles (thumbnails of the model input windows)
# ---------------------------------------------------------------------------


def tile_name(full_key: str) -> str:
    return hashlib.sha1(full_key.encode()).hexdigest()[:20] + ".jpg"


class TileStore:
    """Thumbnails (JPEG bytes) and the point's position in each, in one SQLite
    file. Safe to share between processes (e.g. ``curate`` and the review grid);
    a lock serialises the threads of one process."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS tiles (name TEXT PRIMARY KEY, "
                "data BLOB NOT NULL, x REAL, y REAL)")
            self._db.commit()

    def get(self, name: str) -> Optional[Tuple[bytes, float, float]]:
        with self._lock:
            row = self._db.execute(
                "SELECT data, x, y FROM tiles WHERE name = ?", (name,)).fetchone()
        return None if row is None else (bytes(row[0]), row[1], row[2])

    def positions(self, names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
        """``name -> (x, y)`` of the stored ones among ``names``."""
        out: Dict[str, Tuple[float, float]] = {}
        names = list(dict.fromkeys(names))
        with self._lock:
            for s in range(0, len(names), 500):
                part = names[s:s + 500]
                q = ",".join("?" * len(part))
                for n, x, y in self._db.execute(
                        f"SELECT name, x, y FROM tiles WHERE name IN ({q})", part):
                    out[n] = (x, y)
        return out

    def put_many(self, items: Iterable[Tuple[str, bytes, float, float]]) -> None:
        items = list(items)
        if not items:
            return
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO tiles (name, data, x, y) VALUES (?, ?, ?, ?)",
                items)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def tile_store(folder: str) -> TileStore:
    """The thumbnail store of a classifier folder."""
    return TileStore(os.path.join(curation_dir(folder), settings.CURATION_TILES_FILE))


def encode_tile(window: np.ndarray) -> bytes:
    """A model input window as a JPEG thumbnail (``CURATION_TILE_PX``)."""
    px = settings.CURATION_TILE_PX
    buf = BytesIO()
    Image.fromarray(window).resize((px, px), Image.BILINEAR).save(
        buf, "JPEG", quality=settings.CURATION_TILE_QUALITY)
    return buf.getvalue()


def _tile_writer(store: TileStore, names: Sequence[str], window: int):
    """``on_windows`` callback saving each window as a thumbnail in ``store``;
    also records the point's position in the thumbnail (``pos``)."""
    px = settings.CURATION_TILE_PX
    pos = np.full((len(names), 2), np.nan, np.float32)

    def on_windows(idx, wins, xy):
        items = []
        for j, i in enumerate(idx):
            pos[i] = xy[j] * px / window
            items.append((names[i], encode_tile(wins[j]),
                          round(float(pos[i][0]), 1), round(float(pos[i][1]), 1)))
        store.put_many(items)

    return on_windows, pos


def render_tiles(
    store: TileStore, names: Sequence[str], paths: Sequence[str],
    xs: Sequence[float], ys: Sequence[float], scales: Sequence[float],
    window: int, n_jobs: int = -1,
) -> int:
    """Render thumbnails without the backbone (e.g. after the store was
    deleted); returns how many were written."""
    from substrata import pointfeatures as pf  # lazy

    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        groups.setdefault(p, []).append(i)
    on_win, _pos = _tile_writer(store, names, window)
    jobs = (os.cpu_count() or 1) if n_jobs in (None, -1) else max(1, int(n_jobs))

    def load(item):
        p, idx = item
        res = pf.image_windows(p, np.array([[xs[i], ys[i]] for i in idx], float),
                               np.array([scales[i] for i in idx], float), window)
        if res is not None:
            on_win(np.asarray(idx), res[0], res[1])
            return len(idx)
        return 0

    with ThreadPoolExecutor(jobs) as ex:
        return sum(ex.map(load, groups.items()))


# ---------------------------------------------------------------------------
# Scores of the training points
# ---------------------------------------------------------------------------


def ring_offsets(radius: float = settings.CURATION_RING_RADIUS_PX,
                 n: int = settings.CURATION_RING_N) -> np.ndarray:
    """``(n, 2)`` positions on a circle around a point (model px)."""
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(a), np.sin(a)]).astype(np.float32) * radius


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _l2(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def _knn(query: np.ndarray, base: np.ndarray, k: int,
         q_group: Optional[np.ndarray] = None, b_group: Optional[np.ndarray] = None,
         chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Top-``k`` cosine neighbours of each query row in ``base`` (L2-normalised
    rows), skipping base rows of the same group (e.g. the same photo).

    Returns:
        ``(similarities (nq, k), indices (nq, k))``; -inf / -1 where fewer.
    """
    k = max(1, min(k, len(base)))
    try:
        import torch  # lazy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        torch, dev = None, "cpu"
    sims = np.full((len(query), k), -np.inf, np.float32)
    idx = np.full((len(query), k), -1, np.int64)
    if torch is not None:
        b = torch.from_numpy(np.ascontiguousarray(base, np.float32)).to(dev)
        bg = None if b_group is None else torch.from_numpy(np.asarray(b_group)).to(dev)
        for s in range(0, len(query), chunk):
            q = np.ascontiguousarray(query[s:s + chunk], np.float32)
            q = torch.from_numpy(q).to(dev)
            sim = q @ b.T
            if bg is not None:
                qg = torch.from_numpy(np.asarray(q_group[s:s + chunk])).to(dev)
                sim[qg[:, None] == bg[None, :]] = -float("inf")
            v, i = sim.topk(k, dim=1)
            sims[s:s + len(q)] = v.cpu().numpy()
            idx[s:s + len(q)] = i.cpu().numpy()
    else:  # pragma: no cover
        for s in range(0, len(query), chunk):
            sim = query[s:s + chunk] @ base.T
            if b_group is not None:
                sim[q_group[s:s + chunk, None] == b_group[None, :]] = -np.inf
            i = np.argsort(-sim, axis=1)[:, :k]
            sims[s:s + len(sim)] = np.take_along_axis(sim, i, 1)
            idx[s:s + len(sim)] = i
    idx[~np.isfinite(sims)] = -1
    return sims, idx


def _cv_probabilities(
    z: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: Sequence[str],
    c: float, folds: int, seed: int = 0,
) -> np.ndarray:
    """Out-of-fold class probabilities ``(n, K)`` of a class-balanced logistic
    probe, folds grouped (e.g. by photo) so neighbouring points of one photo
    never predict each other."""
    from sklearn.linear_model import LogisticRegression  # lazy
    from sklearn.model_selection import GroupKFold

    out = np.full((len(y), len(classes)), np.nan, np.float32)
    col = {c_: i for i, c_ in enumerate(classes)}
    n_groups = len(set(groups.tolist()))
    folds = max(2, min(folds, n_groups))
    rng = np.random.default_rng(seed)
    # GroupKFold is deterministic by group size; shuffle the group labels so
    # folds are not ordered by photo
    uniq = np.unique(groups)
    perm = dict(zip(uniq.tolist(), rng.permutation(len(uniq)).tolist()))
    g = np.array([perm[v] for v in groups.tolist()])
    for tr, te in GroupKFold(folds).split(z, y, g):
        if len(set(y[tr].tolist())) < 2:
            continue
        m = z[tr].mean(0)
        s = z[tr].std(0) + 1e-6
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
        lr.fit((z[tr] - m) / s, y[tr])
        p = lr.predict_proba((z[te] - m) / s)
        full = np.zeros((len(te), len(classes)), np.float32)
        for j, k in enumerate(lr.classes_):
            full[:, col[str(k)]] = p[:, j]
        out[te] = full
    return out


def _load_npz(path: str) -> Tuple[Dict[str, int], Optional[np.ndarray]]:
    if not os.path.isfile(path):
        return {}, None
    try:
        z = np.load(path, allow_pickle=False)
        return {k: i for i, k in enumerate(z["keys"].tolist())}, z["feats"]
    except (OSError, ValueError, KeyError):
        return {}, None


def score_training_points(
    folder: str, clf: Any, ann_path: Optional[str] = None,
    batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1,
    folds: int = settings.CURATION_CV_FOLDS, progress: bool = True,
) -> Dict[str, Any]:
    """Score the training points of a classifier folder for review.

    Writes ``curate/points.csv`` (one row per training point; see
    ``POINT_COLUMNS``), ``curate/meta.json`` and a thumbnail per point under
    ``curate/tiles/``. Point features come from the training cache (no new
    extraction when ``train`` ran on the same points); the neighbourhood ring
    is read in the same windows, extracted once and cached.

    Scores (NaN where they do not apply):
      * ``p_label``, ``cv_pred``, ``cv_conf``: cross-validated probability of
        the point's own class and the predicted class (folds grouped by photo);
      * ``ring_agree``: mean probability of the point's class at
        ``CURATION_RING_N`` positions ``CURATION_RING_RADIUS_PX`` around it -
        low for points on boundaries and small objects;
      * ``ring_sim``: mean cosine similarity of the point to its ring;
      * ``knn_label``: share of its ``CURATION_KNN`` nearest neighbours (other
        photos) with the same class;
      * ``novelty``: 1 - mean similarity to those neighbours.

    Returns:
        Summary dict (counts, cross-validated accuracy and macro-F1, paths).
    """
    from substrata import pointfeatures as pf  # lazy (heavy)

    ann_path = ann_path or os.path.join(folder, settings.TRAIN_ANNOTATIONS_FILE)
    if not os.path.isfile(ann_path):
        raise ValueError(f"Training annotations not found: {ann_path} (run "
                         "`substrata train` in this folder first).")
    rows = pf.read_training_points(ann_path, None)
    if not rows:
        raise ValueError(f"No training points in {ann_path}.")
    n = len(rows)
    cls = np.array([clf.label_map.get(r["label"].strip()) or "" for r in rows])
    out_dir = curation_dir(folder)
    os.makedirs(out_dir, exist_ok=True)
    tiles = tile_store(folder)

    # Scales and centre features exactly as training (cache hits)
    gsd = None
    if clf.target_mm_px:
        gsd = pf._fill_gsd(rows, pf.resolve_training_gsd(rows, os.path.join(
            folder, settings.POINT_FEATURES_DIR, "gsd_cache.json")))
    scales = clf.scales_for(gsd, n)
    keys = [pf.row_key(r) for r in rows]
    paths = [r["cam_filepath"] for r in rows]
    xs = [r["_x"] for r in rows]
    ys = [r["_y"] for r in rows]
    model_cache: Dict[str, Any] = {}

    def loader():
        if "m" not in model_cache:
            model_cache["m"] = clf.backbone_model()
        return model_cache["m"]

    feats, ok = pf.cached_features(
        keys, paths, xs, ys, scales,
        pf._cache_file(folder, clf.backbone, clf.window, "points"), loader,
        clf.window, batch_size, n_jobs, desc="Point features",
        offsets=pf._train_offsets(keys, clf.window, clf.meta.get("train_offsets")),
    )

    # Ring features (cached) and thumbnails, in one pass over the photos
    ring = ring_offsets()
    full_keys = [f"{k}|{s:.5f}" for k, s in zip(keys, scales)]
    names = [tile_name(k) for k in full_keys]
    ring_path = os.path.join(
        out_dir, f"ring_{clf.backbone}_w{clf.window}"
        f"_r{settings.CURATION_RING_RADIUS_PX}n{settings.CURATION_RING_N}.npz")
    have, old = _load_npz(ring_path)
    m_ring = len(ring)
    dim = int(feats.shape[1])
    ring_feats = np.zeros((n, m_ring, dim), np.float32)
    ring_ok = np.zeros(n, bool)
    if old is not None and old.shape[1] == m_ring * dim:
        for i, k in enumerate(full_keys):
            j = have.get(k)
            if j is not None:
                ring_feats[i] = old[j].reshape(m_ring, dim)
                ring_ok[i] = True
    stored = tiles.positions(names)
    tile_ok = np.array([nm in stored for nm in names])
    todo = np.nonzero(ok & ~(ring_ok & tile_ok))[0]
    if len(todo):
        print(f"Neighbourhoods and thumbnails: {n - len(todo)} cached, "
              f"reading {len(todo)}")
        on_win, _pos = _tile_writer(tiles, [names[i] for i in todo], clf.window)
        rf, rok = pf.extract_point_features(
            [paths[i] for i in todo], [xs[i] for i in todo], [ys[i] for i in todo],
            scales[todo], loader(), clf.window, batch_size=batch_size,
            n_jobs=n_jobs, desc="Neighbourhoods", progress=progress,
            sample_offsets=ring, on_windows=on_win,
        )
        ring_feats[todo] = rf
        ring_ok[todo] = rok
        stored = tiles.positions(names)
    keep = np.nonzero(ring_ok)[0]
    np.savez(ring_path, keys=np.array([full_keys[i] for i in keep]),
             feats=ring_feats[keep].reshape(len(keep), -1).astype(np.float16))

    # Standardise (per camera domain) and score
    domains = [(r.get("domain") or "").strip() for r in rows]
    z = clf.standardise_rows(feats, domains)
    zr = clf.standardise_rows(ring_feats, domains)
    use = ok & (cls != "")
    classes = list(clf.classes)
    col = {c: i for i, c in enumerate(classes)}
    photo = np.array([os.path.basename(p) for p in paths])

    p_label = np.full(n, np.nan, np.float32)
    cv_pred = np.array([""] * n, dtype=object)
    cv_conf = np.full(n, np.nan, np.float32)
    cv_summary: Dict[str, Any] = {}
    if use.sum() >= 10 and len(set(cls[use].tolist())) >= 2:
        print(f"Cross-validated probabilities ({folds} folds by photo, "
              f"{int(use.sum())} points) ...")
        c = float(clf.meta.get("probe_C") or 0.01)
        probs = _cv_probabilities(z[use], cls[use], photo[use], classes, c, folds)
        idx = np.nonzero(use)[0]
        good = np.isfinite(probs).all(1)
        for j, i in enumerate(idx):
            if not good[j]:
                continue
            k = int(np.argmax(probs[j]))
            cv_pred[i] = classes[k]
            cv_conf[i] = probs[j, k]
            if cls[i] in col:
                p_label[i] = probs[j, col[cls[i]]]
        ev = pf.evaluate_predictions(cls[idx[good]].tolist(),
                                     [cv_pred[i] for i in idx[good]])
        cv_summary = {k: round(float(ev[k]), 4)
                      for k in ("accuracy", "balanced_accuracy", "macro_f1")}
        cv_summary["folds"] = folds

    ring_agree = np.full(n, np.nan, np.float32)
    ring_sim = np.full(n, np.nan, np.float32)
    rr = ring_ok & ok
    if rr.any():
        pr = _softmax(zr[rr] @ clf.weights.T + clf.bias)  # (m, M, K)
        ring_sim[rr] = (_l2(zr[rr]) * _l2(z[rr])[:, None, :]).sum(-1).mean(1)
        for j, i in enumerate(np.nonzero(rr)[0]):
            if cls[i] in col:
                ring_agree[i] = pr[j, :, col[cls[i]]].mean()

    knn_label = np.full(n, np.nan, np.float32)
    novelty = np.full(n, np.nan, np.float32)
    if ok.sum() > 1:
        base = np.nonzero(ok)[0]
        zn = _l2(z[base])
        _, gid = np.unique(photo[base], return_inverse=True)
        sims, nn = _knn(zn, zn, settings.CURATION_KNN, gid, gid)
        for j, i in enumerate(base):
            good = nn[j] >= 0
            if not good.any():
                continue
            novelty[i] = 1.0 - float(sims[j][good].mean())
            if cls[i]:
                nb = cls[base[nn[j][good]]]
                nb = nb[nb != ""]
                if len(nb):
                    knn_label[i] = float((nb == cls[i]).mean())

    def fmt(v, d=4):
        return "" if v is None or not np.isfinite(v) else f"{float(v):.{d}f}"

    table_path = os.path.join(out_dir, settings.CURATION_POINTS_FILE)
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POINT_COLUMNS)
        for i, r in enumerate(rows):
            tp = stored.get(names[i])
            w.writerow([
                i, (r.get("id") or "").strip(), row_point_key(r), r["label"].strip(),
                cls[i], pf.SPLIT_NAMES[r["_split"]], paths[i], f"{xs[i]:.1f}",
                f"{ys[i]:.1f}", f"{scales[i]:.5f}", domains[i], int(bool(ok[i])),
                names[i] if tp else "", tp[0] if tp else "", tp[1] if tp else "",
                cv_pred[i], fmt(cv_conf[i]), fmt(p_label[i]), fmt(ring_agree[i]),
                fmt(ring_sim[i]), fmt(knn_label[i]), fmt(novelty[i]),
            ])

    clean = use & (p_label >= settings.CURATION_CLEAN_P) & (
        ring_agree >= settings.CURATION_CLEAN_RING)
    summary = {
        "n_points": n, "n_ok": int(ok.sum()), "n_mapped": int(use.sum()),
        "n_clean": int(clean.sum()),
        "n_suspect": int((use & (p_label < settings.CURATION_SUSPECT_P)).sum()),
        "n_mixed": int((use & (ring_agree < settings.CURATION_MIXED_RING)).sum()),
        "cv": cv_summary, "table": table_path, "tiles": tiles.path,
    }
    meta = {
        "created": _now(), "annotations": os.path.abspath(ann_path),
        "classifier_classes": classes, "label_map": dict(clf.label_map),
        "window": clf.window, "tile_px": settings.CURATION_TILE_PX,
        "target_mm_px": clf.target_mm_px,
        "ring": {"radius_px": settings.CURATION_RING_RADIUS_PX,
                 "n": settings.CURATION_RING_N},
        "knn": settings.CURATION_KNN,
        "clean": {"p_label": settings.CURATION_CLEAN_P,
                  "ring_agree": settings.CURATION_CLEAN_RING},
        **{k: v for k, v in summary.items() if k not in ("table", "tiles")},
    }
    with open(os.path.join(out_dir, settings.CURATION_META_FILE), "w") as f:
        json.dump(meta, f, indent=1)
    return summary


# ---------------------------------------------------------------------------
# Pools: unlabelled points mined from photos
# ---------------------------------------------------------------------------


def _grid_points(w: int, h: int, n: int, margin_x: float, margin_y: float,
                 rng: np.random.Generator) -> List[Tuple[float, float]]:
    """``n`` points on a jittered grid inside the margins."""
    x0, x1 = margin_x, max(margin_x + 1, w - margin_x)
    y0, y1 = margin_y, max(margin_y + 1, h - margin_y)
    aspect = (x1 - x0) / max(1e-6, (y1 - y0))
    nx = max(1, int(round(np.sqrt(n * aspect))))
    ny = max(1, int(np.ceil(n / nx)))
    cells = [(i, j) for j in range(ny) for i in range(nx)]
    pick = rng.choice(len(cells), min(n, len(cells)), replace=False)
    out = []
    for c in sorted(pick):
        i, j = cells[c]
        # rounded as stored in the pool CSV, so keys computed from either agree
        out.append((round(float(x0 + (i + rng.uniform()) * (x1 - x0) / nx), 1),
                    round(float(y0 + (j + rng.uniform()) * (y1 - y0) / ny), 1)))
    return out


def domain_statistics(folder: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Feature statistics of each pool's camera domain (``domain -> (mean,
    std)``), from the pools of a classifier folder."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for csv_path in pool_csvs(folder):
        stem = os.path.splitext(csv_path)[0]
        try:
            with open(stem + ".json") as f:
                meta = json.load(f)
            z = np.load(stem + ".npz", allow_pickle=False)
            out.setdefault(meta["domain"], (z["mean"].astype(np.float32),
                                            z["std"].astype(np.float32)))
        except (OSError, ValueError, KeyError):
            continue
    return out


def reference_width(folder: str, clf: Any) -> Optional[float]:
    """Median width (px) of the classifier's training photos (its main camera
    domain): the classifier's ``meta["ref_width"]``, else read from the
    folder's training annotations (classifiers trained before it was kept)."""
    from substrata import pointfeatures as pf  # lazy (heavy)

    ref = (clf.meta or {}).get("ref_width")
    if ref:
        return float(ref)
    ann = os.path.join(folder, settings.TRAIN_ANNOTATIONS_FILE)
    if not os.path.isfile(ann):
        return None
    paths = [r["cam_filepath"] for r in read_table(ann)
             if r.get("cam_filepath") and not (r.get("domain") or "").strip()]
    return pf._median_image_width(paths)


def read_pool_points(
    csv_path: str, image_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Points for a pool from a points CSV (``cam_filepath``, ``cam_x``,
    ``cam_y``; optional ``id``, ``label``) or a CoralNet export (photos found
    under ``image_dir``, default the CSV's folder), as with image-only
    ``substrata predict --points``.

    Returns:
        ``(points, info)``: dicts with ``path``, ``x``, ``y``, ``id`` and
        ``label`` (the CSV's, kept as the pool's ``given_label``) and ``gsd``
        (its ``cam_gsd_mm``, else None) on readable photos, and counts of what
        was read and skipped.
    """
    import tempfile

    from substrata import classification  # lazy

    with open(csv_path, newline="") as f:
        header = set(next(csv.reader(f), []))
    info: Dict[str, Any] = {"source": os.path.abspath(csv_path), "coralnet": False}
    src = csv_path
    tmp = None
    if {"Name", "Row", "Column"} <= header:
        folder = image_dir or os.path.dirname(os.path.abspath(csv_path))
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp.close()
        conv = classification.convert_coralnet_csv(
            csv_path, classification.index_image_dir(folder), tmp.name)
        info.update(coralnet=True, image_dir=folder, n_points=conv["n_points"],
                    n_images=conv["n_images"], n_images_found=conv["n_images_found"])
        src = tmp.name
    try:
        rows = read_table(src)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)
    base = os.path.dirname(os.path.abspath(csv_path))
    out, missing = [], 0
    exists: Dict[str, bool] = {}
    for r in rows:
        p = (r.get("cam_filepath") or "").strip()
        try:
            x, y = float(r["cam_x"]), float(r["cam_y"])
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        if p and not os.path.isabs(p) and not os.path.isfile(p):
            p = os.path.join(base, p)
        if p not in exists:
            exists[p] = bool(p) and os.path.isfile(p)
        if not exists[p] or not np.isfinite(x) or not np.isfinite(y):
            missing += 1
            continue
        try:
            g = float(r.get("cam_gsd_mm") or "nan")
        except ValueError:
            g = float("nan")
        out.append({"path": p, "x": round(x, 1), "y": round(y, 1),
                    "id": (r.get("id") or "").strip(),
                    "label": (r.get("label") or "").strip(),
                    "gsd": g if np.isfinite(g) and g > 0 else None})
    info.update(n_rows=len(rows), n_used=len(out), n_skipped=missing)
    return out, info


def mine_pool(
    folder: str, clf: Any, images: Sequence[str], name: str,
    gsd_mm: Optional[float] = None, domain: Optional[str] = None,
    n_images: int = settings.CURATION_MINE_IMAGES,
    per_image: int = settings.CURATION_MINE_PER_IMAGE,
    n_clusters: int = settings.CURATION_MINE_CLUSTERS, seed: int = 0,
    resample: bool = False, batch_size: int = settings.POINT_BATCH_SIZE,
    n_jobs: int = -1, progress: bool = True,
    points: Optional[Sequence[Dict[str, Any]]] = None,
    points_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample unlabelled points from photos, cluster them and score novelty.

    Points sit on a jittered grid in a random sample of the photos, or are
    given (``points``, from :func:`read_pool_points`; their labels are kept as
    ``given_label``, which a reviewer accepts or changes). Each photo is
    resampled to the classifier's resolution: ``gsd_mm`` / its target; without
    a ground resolution the window covers the share of the photo's width it
    covers on the training photos (:func:`substrata.pointfeatures.photo_scale`,
    as ``substrata train`` resamples the pool's points). Their features are
    standardised with their own statistics (a camera domain, like ``substrata
    calibrate``), clustered (PCA + k-means), predicted with the classifier and
    compared with the training points (novelty = 1 - mean similarity to the
    nearest training points).

    Writes ``curate/pools/<name>.csv`` (a points CSV: label empty, plus
    ``POOL_EXTRA_COLUMNS``), ``<name>.npz`` (features and domain statistics)
    and ``<name>.json``. Re-running with the same sampling only re-clusters.

    Returns:
        The pool's meta dict (with per-cluster summaries).
    """
    from substrata import pointfeatures as pf  # lazy (heavy)
    from sklearn.cluster import KMeans  # lazy
    from sklearn.decomposition import PCA

    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    if not name:
        raise ValueError("A pool needs a name (--name).")
    domain = domain or name
    pools_dir = os.path.join(curation_dir(folder), settings.CURATION_POOLS_DIR)
    os.makedirs(pools_dir, exist_ok=True)
    stem = os.path.join(pools_dir, name)
    tiles = tile_store(folder)
    ref_w = None if gsd_mm and clf.target_mm_px else reference_width(folder, clf)
    point_gsd = [q.get("gsd") for q in points] if points is not None else []
    all_gsd = bool(clf.target_mm_px) and (bool(gsd_mm) or (
        bool(point_gsd) and all(point_gsd)))
    if not all_gsd:
        if ref_w:
            print(f"Pool {name}: points without a ground resolution (--gsd-mm, a "
                  f"cam_gsd_mm column) on photos of another camera are resampled so the "
                  f"window covers the share of their width it covers on the training "
                  f"photos ({ref_w:.0f} px wide at scale {clf.scale_factor:.3f})")
        else:
            logger.warning("No ground resolution or training photo width: the pool is "
                           "sampled at the classifier's fixed scale %.3f.", clf.scale_factor)
    widths: Dict[str, Optional[int]] = {}

    def scale_of(p, g=None):
        if p not in widths:
            widths.update(pf._image_widths([p]))
        return pf.photo_scale(clf, g or gsd_mm, widths.get(p), ref_w)

    scale = pf.photo_scale(clf, gsd_mm)
    sampling = {"images": len(images), "n_images": n_images, "per_image": per_image,
                "seed": seed, "scale": round(scale, 6) if not ref_w else "width",
                "window": clf.window, "backbone": clf.backbone}
    if points is not None:
        sampling = {"points": (points_info or {}).get("source"), "n_points": len(points),
                    "scale": sampling["scale"], "window": clf.window,
                    "backbone": clf.backbone}
    if ref_w:  # (absent otherwise, so pools mined before it was added are re-used)
        sampling["ref_width"] = round(float(ref_w), 1)
    meta_old: Dict[str, Any] = {}
    if os.path.isfile(stem + ".json"):
        with open(stem + ".json") as f:
            meta_old = json.load(f)

    have_files = os.path.isfile(stem + ".npz") and os.path.isfile(stem + ".csv")
    reuse = not resample and meta_old.get("sampling") == sampling and have_files
    if reuse:
        z = np.load(stem + ".npz", allow_pickle=False)
        feats = z["feats"].astype(np.float32)
        old_rows = read_table(stem + ".csv")
        pts = [(r["cam_filepath"], float(r["cam_x"]), float(r["cam_y"]))
               for r in old_rows]
        scales = [float(r.get("scale") or 0) or scale_of(r["cam_filepath"])
                  for r in old_rows]
        extra = [((r.get("id") or "").strip(), (r.get("given_label") or "").strip())
                 for r in old_rows]
        gsds = [float(r["cam_gsd_mm"]) if r.get("cam_gsd_mm") else None for r in old_rows]
        tile_xy = [(r.get("tile"), r.get("tile_x"), r.get("tile_y")) for r in old_rows]
        print(f"Pool {name}: re-using {len(pts)} sampled points")
        names = [t[0] for t in tile_xy]
        stored = tiles.positions([nm for nm in names if nm])
        miss = [i for i, nm in enumerate(names) if nm and nm not in stored]
        if miss:
            print(f"Pool {name}: rendering {len(miss)} missing thumbnail(s)")
            render_tiles(tiles, [names[i] for i in miss], [pts[i][0] for i in miss],
                         [pts[i][1] for i in miss], [pts[i][2] for i in miss],
                         [scales[i] for i in miss], clf.window, n_jobs)
    else:
        cand: List[Tuple[str, float, float]] = []
        cand_extra: List[Tuple[str, str]] = []
        if points is not None:
            cand = [(q["path"], float(q["x"]), float(q["y"])) for q in points]
            cand_extra = [(q.get("id") or "", q.get("label") or "") for q in points]
            cand_gsd = [q.get("gsd") or gsd_mm for q in points]
            n_photos = len({p for p, _x, _y in cand})
        else:
            rng = np.random.default_rng(seed)
            imgs = sorted(set(images))
            pick = [imgs[i] for i in sorted(rng.choice(len(imgs), min(n_images, len(imgs)),
                                                       replace=False))]
            for p in pick:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except OSError:
                    continue
                widths[p] = w
                half = clf.window / scale_of(p) / 2
                mx, my = min(half, w / 2 - 1), min(half, h / 2 - 1)
                cand += [(p, x, y) for x, y in _grid_points(w, h, per_image, mx, my, rng)]
            cand_extra = [("", "")] * len(cand)
            cand_gsd = [gsd_mm] * len(cand)
            n_photos = len(pick)
        if not cand:
            raise ValueError("No readable photos to sample.")
        cand_scales = np.array([scale_of(p, g) for (p, _x, _y), g in zip(cand, cand_gsd)],
                               float)
        lo, hi = float(cand_scales.min()), float(cand_scales.max())
        span = f"{lo:.3f}" if hi - lo < 1e-4 else f"{lo:.3f}-{hi:.3f}"
        print(f"Pool {name}: {len(cand)} points in {n_photos} photo(s) at scale {span}")
        full = [f"{name}|{point_key(p, x, y)}|{s:.5f}" for (p, x, y), s in zip(cand, cand_scales)]
        names = [tile_name(k) for k in full]
        on_win, pos = _tile_writer(tiles, names, clf.window)
        f_all, ok = pf.extract_point_features(
            [p for p, _x, _y in cand], [x for _p, x, _y in cand],
            [y for _p, _x, y in cand], cand_scales, clf.backbone_model(),
            clf.window, batch_size=batch_size, n_jobs=n_jobs, desc=f"Pool {name}",
            progress=progress, on_windows=on_win,
        )
        keep = np.nonzero(ok)[0]
        pts = [cand[i] for i in keep]
        scales = [float(cand_scales[i]) for i in keep]
        extra = [cand_extra[i] for i in keep]
        gsds = [cand_gsd[i] for i in keep]
        feats = f_all[keep]
        tile_xy = [(names[i], f"{pos[i][0]:.1f}", f"{pos[i][1]:.1f}") for i in keep]
    if len(feats) < 2:
        raise ValueError("Too few readable points in the pool.")

    mean, std = pf.feature_statistics(feats)
    zp = (feats - mean) / std
    probs = _softmax(zp @ clf.weights.T + clf.bias)
    pred = np.array(clf.classes)[probs.argmax(1)]
    conf = probs.max(1)

    # Novelty against the training points (their own statistics)
    novelty = np.full(len(feats), np.nan, np.float32)
    have, tf = _load_npz(pf._cache_file(folder, clf.backbone, clf.window, "points"))
    if tf is not None and len(tf):
        zt = _l2((tf.astype(np.float32) - clf.source_mean) / clf.source_std)
        sims, _ = _knn(_l2(zp), zt, settings.CURATION_KNN)
        novelty = 1.0 - np.where(np.isfinite(sims), sims, np.nan).mean(1)

    # Clusters, largest first; within a cluster, distance to its centre
    k = int(max(2, min(n_clusters, len(feats) // 5 or 2)))
    zn = _l2(zp)
    dims = int(min(settings.CURATION_MINE_PCA, zn.shape[1], len(zn) - 1))
    zr = PCA(dims, random_state=seed).fit_transform(zn) if dims >= 2 else zn
    km = KMeans(k, n_init=3, random_state=seed).fit(zr)
    order = np.argsort(-np.bincount(km.labels_, minlength=k), kind="stable")
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    clusters = remap[km.labels_]
    dist = np.linalg.norm(zr - km.cluster_centers_[km.labels_], axis=1)

    np.savez(stem + ".npz", feats=feats.astype(np.float16), mean=mean, std=std)
    cols = list(settings.TRAIN_ANN_COLUMNS) + [c for c in POOL_EXTRA_COLUMNS
                                               if c not in settings.TRAIN_ANN_COLUMNS]
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (p, x, y) in enumerate(pts):
            row = {c: "" for c in cols}
            for axis in "xyz":
                row[f"orig_{axis}"] = row[f"world_{axis}"] = "nan"
            tile, tx, ty = tile_xy[i]
            pid, given = extra[i]
            row.update({
                "id": pid or f"{name}_{i}", "cam_filepath": p, "cam_x": f"{x:.1f}",
                "cam_y": f"{y:.1f}",
                "split_key": os.path.splitext(os.path.basename(p))[0],
                "cam_gsd_mm": "" if not gsds[i] else f"{gsds[i]:.4f}", "domain": domain,
                "key": point_key(p, x, y),
                "cluster": int(clusters[i]), "cluster_dist": f"{dist[i]:.4f}",
                "pred_label": pred[i], "pred_conf": f"{conf[i]:.4f}",
                "novelty": "" if not np.isfinite(novelty[i]) else f"{novelty[i]:.4f}",
                "tile": tile or "", "tile_x": tx or "", "tile_y": ty or "",
                "scale": f"{scales[i]:.5f}", "given_label": given,
            })
            w.writerow(row)

    summaries = []
    for c in range(k):
        sel = clusters == c
        cnt = Counter(pred[sel].tolist())
        top, top_n = cnt.most_common(1)[0]
        summaries.append({
            "cluster": c, "n": int(sel.sum()), "pred": top,
            "pred_share": round(top_n / sel.sum(), 3),
            "mean_conf": round(float(conf[sel].mean()), 3),
            "novelty": None if not np.isfinite(novelty[sel]).any()
            else round(float(np.nanmean(novelty[sel])), 4),
            "n_photos": len({pts[i][0] for i in np.nonzero(sel)[0]}),
        })
    meta = {
        "name": name, "domain": domain, "created": _now(), "sampling": sampling,
        "gsd_mm": gsd_mm, "n_points": len(pts), "n_clusters": k,
        "n_given_labels": sum(1 for _pid, g in extra if g),
        "classes": list(clf.classes), "clusters": summaries,
    }
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    meta["csv"] = stem + ".csv"
    return meta
