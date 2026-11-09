# Stratified PLY format (v1)

A **stratified PLY** is an ordinary binary PLY whose vertex records have been
*reordered* (never changed), plus a small sidecar index. The order makes three
things cheap:

1. **Overview**: the first *N* records are a uniform random subsample of the whole
   cloud, when *N* is a round boundary (see below). A decimated cloud is then just a
   prefix read, and no separate `_dec50M.ply` is needed.
2. **Regions at any detail**: the cloud is split into spatially compact *clusters*.
   Each cluster, at each detail level, is stored as a few contiguous byte ranges. A box
   can therefore be read at any density without scanning the file, which also works
   over HTTP range requests.
3. **Compatibility**: every record is byte-identical to the input, so any PLY reader
   still loads the full cloud (only the point order differs).

Reference implementation: `src/substrata/stratified.py` in substrata (`StratifiedPly` is
the reader, `stratify_ply` the writer). Build one with `substrata stratify`. Inspect one with
`substrata head` (prints the clusters, rounds and prefix sizes).

---

## 1. Files

| File | What |
| --- | --- |
| `<root>.ply` | Binary PLY, records reordered (section 2). |
| `<root>.strat.npz` | The index (section 3). Same folder and stem: replace the `.ply` extension. |

The PLY on its own is a valid, complete cloud. The index is what makes it *stratified*.

## 2. The PLY

- **Supported input:** `binary_little_endian` or `binary_big_endian`, with a single
  non-empty element, `vertex`. Vertex properties must be scalar; `list` properties
  aren't supported. Other elements may be declared, but only with count 0. ASCII PLY is
  not supported.
- **Header:** the input header, plus one marker line inserted directly after the `format`
  line:
  ```
  ply
  format binary_little_endian 1.0
  comment substrata stratified v1
  element vertex 579428108
  property float x
  ...
  end_header
  ```
  The marker is only a hint for humans and tools. Section 4 is the real test.
- **Records:** fixed size (the sum of the property sizes), in the PLY's endianness,
  starting at byte `header_len`. They're a permutation of the input records. The
  properties are kept as in the input; `x`, `y`, `z` are required.

## 3. The index (`.strat.npz`)

This is a NumPy `.npz` file: an uncompressed ZIP archive (`np.savez`, members stored, not
deflated). It holds one `.npy` member per key below; `version` is stored as
`version.npy`.

- **`.npy` format:** each member is in NPY format 1.0. That is the magic string
  `\x93NUMPY`, the bytes `\x01\x00`, a uint16 (little-endian) header length, then an
  ASCII Python-dict header such as `{'descr': '<i8', 'fortran_order': False, 'shape':
  (13, 472), }`, then the raw C-order data.
- **Outside Python:** any ZIP library plus a small NPY parser reads it (for example
  `npyjs` in JavaScript or `cnpy` in C++).
- **Scalars:** stored as 0-d arrays (`shape: ()`).

| Key | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `version` | `<i8` | () | Format version, currently `1`. Refuse other values. |
| `n` | `<i8` | () | Number of records (= `element vertex` count). |
| `header_len` | `<i8` | () | PLY header size in bytes, including the `end_header\n` line. |
| `record_size` | `<i8` | () | Bytes per record. |
| `rounds` | `<i8` | () | R, the number of detail levels (rounds). |
| `counts` | `<i8` | (R, C) | Records of cluster `c` in round `r`. |
| `starts` | `<i8` | (R, C) | Record index (not bytes) where that slice starts. |
| `bbox_lo` | `<f8` | (C, 3) | Exact per-cluster minimum x, y, z, in the PLY's own (raw) coordinates. |
| `bbox_hi` | `<f8` | (C, 3) | Exact per-cluster maximum x, y, z. |
| `area` | `<f8` | (C,) | Approximate surface area of each cluster (raw units²), for level-of-detail decisions. |
| `file_size` | `<i8` | () | PLY size in bytes when the index was written. |
| `file_mtime_ns` | `<i8` | () | PLY modification time then (ns). For information only. |

C = number of clusters. Real example (579 M points, 28 B records): R = 13, C = 472,
cluster sizes 0.48 M to 2.0 M points, index 124 KB.

### Layout

The records are ordered by round, then cluster. All of round 0 comes first (clusters in
index order), then all of round 1, and so on:

```
round 0: [c0 | c1 | ... | cC-1]  round 1: [c0 | c1 | ... | cC-1]  ...  round R-1: [...]
```

- **Offsets:** `starts` is the running total of `counts` in that (row-major) order, so
  `starts[0,0] = 0`, and the slice of cluster `c` in round `r` is records
  `[starts[r,c], starts[r,c] + counts[r,c])`.
- **Byte offset of record i:** `header_len + i * record_size`.
- **Round ends:** round `r` ends at record `round_ends[r] = sum(counts[0..r, :])`, and
  `round_ends[R-1] = n`.

### Meaning of the rounds

- **Inside a cluster:** the records were randomly shuffled, then split into rounds. So
  rounds `0..r` of one cluster are a uniform random subset of that cluster.
- **Sizes:** the cumulative size after round `r` is
  `ceil(n_c * 2^(r - R + 1))`, and the last round completes the cluster. Each round
  roughly doubles the density: round 0 is about `n_c / 2^(R-1)` points.
- **Choice of R:** by default substrata picks R so that round 0 has about 256–512 points
  per cluster.
- **Don't recompute the sizes:** always use `counts`, not this formula.

**Prefix reads are uniform only at round boundaries.** The prefix `[0, round_ends[r])`
holds rounds 0..r of every cluster, which is a uniform subsample of the whole cloud. A
prefix that ends mid-round is not uniform. It adds the next round only for the first
clusters in index order, and those are spatially adjacent, so one region becomes denser
than the rest. For a point budget *B*, read up to the largest `round_ends[r] <= B`.
Alternatively, read to the next round end and subsample that uniformly.

### Clusters

- **Shape:** each cluster is a box of grid cells from a kd-split. Clusters don't overlap
  in space, but their `bbox` values are the tight bounds of their actual points, and
  those bounds can overlap slightly.
- **Order:** clusters are numbered in depth-first kd order, so consecutive indices are
  spatially close. Readers should rely only on the `bbox` values, not on this order.
- **Area:** `area` is roughly the number of occupied voxels (at a voxel size of the
  cluster extent / 256) × voxel². `sqrt(area / points_read)` estimates the point
  spacing at a given detail. substrata-explore uses this to choose how many rounds to
  read per cluster for a target on-screen spacing.

## 4. Is this PLY stratified? (detection)

A PLY is stratified when:

1. `<root>.strat.npz` exists, and its `version` is 1; **and**
2. the index matches the PLY: `file_size` equals the PLY's current size, `n` equals the
   `element vertex` count, `header_len` equals the parsed header length, and
   `record_size` equals the computed record size.

Anything else (no index, a mismatch, or a read error) means *not stratified*: treat it as
a plain PLY. Re-saving the PLY in another tool (CloudCompare, Metashape, …) reorders it
and/or changes its header, so the check fails, which is correct.

`file_mtime_ns` is not part of the check, because copies and FAT/exFAT drives change
modification times. Use it only as a warning hint.

## 5. Reading recipes

Pseudo-code, with `idx` = the loaded index and `rec(i, k)` = read `k` records from record
`i` (byte `header_len + i * record_size`):

```text
# Overview: a uniform sample of about B points
r   = max { r : round_ends[r] <= B }          (or r = 0 if none)
pts = rec(0, round_ends[r])                    # one contiguous read

# A cluster at detail level k (rounds 0..k-1): k contiguous reads
for r in 0..k-1:  rec(starts[r,c], counts[r,c])

# Everything inside an axis-aligned box [lo, hi], at detail k (k = R for full)
C_hit = { c : all(bbox_hi[c] >= lo) and all(bbox_lo[c] <= hi) }
for c in C_hit:  read cluster c at detail k, keep points inside [lo, hi]

# Progressive refinement of what you already have: add round k of the visible clusters
for c in visible:  rec(starts[k,c], counts[k,c])          # appends; nothing re-read
```

Notes:
- **Coordinates:** all coordinates, including `bbox_lo`/`bbox_hi`, are the PLY's raw
  coordinates. In a substrata project, apply the project YAML's 4×4 `world_transform` to
  get scaled, oriented world coordinates. Transform your query box into raw coordinates
  first; an oriented box becomes the raw-space AABB of its 8 corners.
- **Web streaming:** every slice, and every complete round prefix, is one contiguous
  byte range. A viewer can stream overview → regions → detail with HTTP `Range`
  requests, using only the ~100 KB index plus the PLY.
- **Merging rounds:** you don't need to deduplicate when you add rounds. Each record
  appears exactly once in the file.

Minimal Python reader without substrata:

```python
import numpy as np

def open_stratified(ply_path):
    idx = dict(np.load(ply_path[:-4] + ".strat.npz"))
    assert int(idx["version"]) == 1
    # dtype from the PLY header (properties in order; '<' or '>' from the format line)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])  # ...build from header
    assert dtype.itemsize == int(idx["record_size"])
    rec = np.memmap(ply_path, dtype=dtype, mode="r",
                    offset=int(idx["header_len"]), shape=(int(idx["n"]),))
    return idx, rec

def read_cluster(idx, rec, c, k=None):
    k = int(idx["rounds"]) if k is None else k
    s, n = idx["starts"][:k, c], idx["counts"][:k, c]
    return np.concatenate([rec[a:a + m] for a, m in zip(s, n)])
```

## 6. Writing one (if you need your own writer)

Readers only depend on sections 2–4. How clusters are chosen is up to the writer. A valid
writer must make sure of five things:

1. The output records are a permutation of the input records, byte for byte. The header
   is the input header plus the marker comment.
2. Each cluster's records are uniformly shuffled before they are split into rounds.
   Cumulative round sizes grow about 2× per round, and the last round completes the
   cluster.
3. The file order is round-major then cluster (as under "Layout"), and `starts` follows
   from `counts`.
4. `bbox_lo`/`bbox_hi` are exact over each cluster's points, and `area` is positive.
5. The index is written after the PLY is final, with its `file_size` and `header_len`.

substrata's writer, for reference:
- **Grid:** a grid of about 16.7 M cells over the bounding box, estimated from samples;
  outlying points are clamped into the edge cells.
- **Clusters:** a kd-split of the per-cell counts into boxes of ≤ `cluster_size` points
  (default 2 M). Each split is along the longest axis at the median.
- **Rounds:** R = `clip(floor(log2(median_cluster / 256)) + 1, 1, 16)`.
- **Shuffle:** a per-cluster shuffle with `numpy.random.default_rng(seed=0)`. This makes
  substrata's output reproducible, but other writers don't need to match it.
- **Modes:** an in-memory mode and a 3-pass streaming mode give identical files. Both
  verify xyz and colour checksums before the result replaces the original.
- **Index for an existing file:** an already-reordered file can't be re-indexed after the
  fact, because the order *is* the format. Run `substrata stratify --force` on the PLY
  instead.

## 7. Versioning

`version` (in the index) and `v1` (in the header comment) change together. They change
only when readers would misread the file. Readers should refuse an unknown `version`, and
can fall back to treating the PLY as plain.
