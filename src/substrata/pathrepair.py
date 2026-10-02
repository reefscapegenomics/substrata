"""Locating and repairing the image filepaths stored for each camera.

Camera image paths are recorded when a project is exported from Metashape and
point at wherever the photos lived at that moment. They go stale as soon as the
archive is moved, remounted, or copied between machines. This module locates the
images again and plans the corrections; it never mutates camera objects and
never writes to disk, so a plan can be discarded for free.

The module is deliberately stdlib-only (plus :mod:`substrata.settings` and the
shared logger) and must not import :mod:`substrata.cameras`, which imports this
module for :func:`path_basename`.
"""

from __future__ import annotations

# Standard Library
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Local Modules
from substrata import settings
from substrata.logging import logger

# Result of an interactive decision: ("pick", path) / ("skip", None) /
# ("abort", None).
Decision = Tuple[str, Optional[str]]
DuplicateCallback = Callable[[str, str, List[str]], Decision]
MissingCallback = Callable[[str, str], Decision]

_SEP_RE = re.compile(r"[\\/]+")


class PathAborted(Exception):
    """Raised when the user aborts an interactive path resolution."""


@dataclass
class PathPlan:
    """Resolved image paths for a set of cameras, before anything is written.

    Attributes:
        stored: Camera id to the filepath held before any change, for reporting.
        changes: Camera id to new filepath, for cameras whose path changes.
        unchanged: Camera id to filepath, for paths already valid on disk.
        skipped: Camera id to stored filepath, for cameras the user skipped.
        missing: Camera id to filepath that does not exist on disk.
        dir_map: Remembered old-directory to new-directory mappings.
        n_duplicate_prompts: Number of duplicate decisions put to the user.
        n_missing_prompts: Number of missing-file decisions put to the user.
    """

    stored: Dict[str, str] = field(default_factory=dict)
    changes: Dict[str, str] = field(default_factory=dict)
    unchanged: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    missing: Dict[str, str] = field(default_factory=dict)
    dir_map: Dict[str, str] = field(default_factory=dict)
    n_duplicate_prompts: int = 0
    n_missing_prompts: int = 0

    @property
    def n_prompted(self) -> int:
        """Total number of decisions that required the user."""
        return self.n_duplicate_prompts + self.n_missing_prompts

    def collisions(self) -> Dict[str, List[str]]:
        """New filepaths claimed by more than one camera.

        Two cameras resolving to the same image means a bad find/replace or a
        bad folder mapping, and must never be written.

        Returns:
            dict[str, list[str]]: New filepath to the sorted camera ids
            claiming it, for filepaths claimed more than once.
        """
        by_path: Dict[str, List[str]] = {}
        for cam_id, path in self.changes.items():
            by_path.setdefault(path, []).append(cam_id)
        return {p: sorted(ids) for p, ids in by_path.items() if len(ids) > 1}


def path_basename(path: str) -> str:
    """Final component of a path, treating ``/`` and ``\\`` as separators.

    Unlike :func:`os.path.basename`, this handles the Windows-style paths
    Metashape stores (e.g. ``D:\\photos\\dive1\\IMG_0001.JPG``) when running on
    POSIX, where ``os.path.basename`` would return the whole string.

    Args:
        path: Path in either POSIX or Windows form.

    Returns:
        str: The final path component, or an empty string for an empty path.
    """
    if not path:
        return ""
    parts = [p for p in _SEP_RE.split(path) if p]
    return parts[-1] if parts else ""


def path_dirname(path: str) -> str:
    """Directory part of a path, separator-agnostic and normalised to ``/``.

    Trailing separators are stripped, so ``D:\\a\\b\\`` and ``D:/a/b`` produce
    the same key. Used as the memo key for old-to-new directory mappings.

    Args:
        path: Path in either POSIX or Windows form.

    Returns:
        str: The directory part, or an empty string if there is none.
    """
    if not path:
        return ""
    parts = [p for p in _SEP_RE.split(path) if p]
    if len(parts) <= 1:
        return ""
    prefix = "/" if path[0] in "\\/" else ""
    return prefix + "/".join(parts[:-1])


def build_image_index(
    root: str,
    exts: Tuple[str, ...] = settings.PATHREPAIR_IMAGE_EXTS,
    skip_dir_names: Tuple[str, ...] = settings.PATHREPAIR_SKIP_DIR_NAMES,
    quiet: bool = False,
) -> Dict[str, List[str]]:
    """Index every image file under a directory tree by lower-cased basename.

    Walks ``root`` once. Directories whose name starts with ``.`` or appears in
    ``skip_dir_names`` are pruned and not descended into. Symlinked directories
    are not followed, so symlink loops cannot hang the walk.

    Keys are the full lower-cased basename rather than only a lower-cased
    extension, so a case-renamed file (``IMG_1.JPG`` to ``img_1.jpg``) is still
    found. Two files differing only in case therefore collide into one entry and
    surface as an ordinary duplicate decision.

    Args:
        root: Directory to search recursively.
        exts: Lower-case file extensions to index.
        skip_dir_names: Directory names to prune.
        quiet: Suppress the indexed-count message.

    Returns:
        dict[str, list[str]]: Lower-cased basename to a sorted list of absolute
        filepaths. A list with more than one entry is a duplicate basename.
    """
    root_abs = os.path.abspath(root)
    index: Dict[str, List[str]] = {}
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(root_abs):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d not in skip_dir_names
        ]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            index.setdefault(name.lower(), []).append(os.path.join(dirpath, name))
            n_files += 1
    for paths in index.values():
        paths.sort()
    logger.info("Indexed %s image file(s) under %s", n_files, root_abs)
    if not quiet:
        print(f"Indexed {n_files} image file(s) under {root_abs}")
    return index


def zero_byte_paths(paths: List[str]) -> List[str]:
    """Filepaths that exist but are zero bytes.

    Reported separately rather than treated as missing: locating an image and
    validating it are different jobs, and a truncated source photo is a problem
    the user should see rather than one this module should silently route
    around.

    Args:
        paths: Filepaths to check.

    Returns:
        list[str]: Those that exist and have a size of zero.
    """
    empty = []
    for path in paths:
        try:
            if os.path.isfile(path) and os.path.getsize(path) == 0:
                empty.append(path)
        except OSError:
            continue
    return empty


def _final_dir(path: str) -> str:
    """Lower-cased final directory component of a path, for disambiguation."""
    return path_basename(path_dirname(path)).lower()


def resolve_image_paths(
    paths: Dict[str, str],
    index: Dict[str, List[str]],
    on_duplicate: Optional[DuplicateCallback] = None,
    on_missing: Optional[MissingCallback] = None,
) -> PathPlan:
    """Resolve each camera's stored path to a file found under the search root.

    Reads only the given mapping and returns a plan; no camera object is touched
    and nothing is written, so an abort costs nothing. Cameras are processed in
    sorted id order, making prompt order and memo population deterministic.

    Resolution order per camera, first hit wins:

    1. The stored path already exists on disk, recorded as unchanged.
    2. A remembered old-to-new directory mapping yields an existing file.
    3. Exactly one index entry for the basename, taken silently.
    4. Several index entries, exactly one of which sits in a directory whose
       final component matches the stored path's, taken silently.
    5. Several index entries, put to ``on_duplicate``.
    6. No index entry, put to ``on_missing``.

    Whenever a camera is resolved by callback or by rule 3 or 4, the mapping
    from its stored directory to the chosen directory is remembered and reused
    by rule 2, so one answer repairs a whole repeated folder pattern. The memo
    is always verified with :func:`os.path.isfile` before use, so a wrong
    mapping cannot fabricate a path; it simply falls through to a prompt.

    Args:
        paths: Camera id to currently stored filepath.
        index: Basename index from :func:`build_image_index`.
        on_duplicate: Called as ``(cam_id, stored_path, candidates)`` when a
            basename matches several files. Returns ``("pick", path)``,
            ``("skip", None)`` or ``("abort", None)``.
        on_missing: Called as ``(cam_id, stored_path)`` when no file matches.
            Same return contract.

    Returns:
        PathPlan: The planned changes.

    Raises:
        PathAborted: If a callback returns ``"abort"``.
        RuntimeError: If a decision is needed but the matching callback is
            ``None`` (a non-interactive context).
    """
    plan = PathPlan(stored=dict(paths))
    for cam_id in sorted(paths):
        stored = paths[cam_id]
        if not stored:
            continue

        # 1. Already valid. Never rewrite a working path, even if the index
        # holds another copy of the same image elsewhere.
        if os.path.isfile(stored):
            plan.unchanged[cam_id] = stored
            continue

        basename = path_basename(stored)
        old_dir = path_dirname(stored)

        # 2. Remembered folder mapping, verified before use.
        mapped_dir = plan.dir_map.get(old_dir)
        if mapped_dir:
            candidate = os.path.join(mapped_dir, basename)
            if os.path.isfile(candidate):
                _record(plan, cam_id, stored, candidate, old_dir)
                continue

        candidates = index.get(basename.lower(), [])

        if len(candidates) == 1:
            _record(plan, cam_id, stored, candidates[0], old_dir)
            continue

        if len(candidates) > 1:
            # 4. Disambiguate by matching final directory component. This
            # removes most duplicate prompts in practice (dive1 vs dive2).
            want = _final_dir(stored)
            same_dir = [c for c in candidates if _final_dir(c) == want]
            if want and len(same_dir) == 1:
                logger.debug(
                    "Camera %s: matched %s by final directory %r",
                    cam_id,
                    basename,
                    want,
                )
                _record(plan, cam_id, stored, same_dir[0], old_dir)
                continue

            if on_duplicate is None:
                raise RuntimeError(
                    f"Camera {cam_id}: {len(candidates)} files named {basename!r} "
                    "found and no way to ask which to use."
                )
            plan.n_duplicate_prompts += 1
            action, chosen = on_duplicate(cam_id, stored, candidates)
        else:
            if on_missing is None:
                raise RuntimeError(
                    f"Camera {cam_id}: no file named {basename!r} found and no "
                    "way to ask where it is."
                )
            plan.n_missing_prompts += 1
            action, chosen = on_missing(cam_id, stored)

        if action == "abort":
            raise PathAborted(f"Aborted while resolving camera {cam_id}.")
        if action == "skip" or not chosen:
            plan.skipped[cam_id] = stored
            continue
        _record(plan, cam_id, stored, chosen, old_dir)

    return plan


def _record(
    plan: PathPlan, cam_id: str, stored: str, chosen: str, old_dir: str
) -> None:
    """Record a resolved path on the plan and remember its folder mapping."""
    if chosen == stored:
        plan.unchanged[cam_id] = stored
        return
    plan.changes[cam_id] = chosen
    new_dir = path_dirname(chosen)
    if old_dir and new_dir and plan.dir_map.get(old_dir) != new_dir:
        plan.dir_map[old_dir] = new_dir


def plan_find_replace(paths: Dict[str, str], find: str, replace: str) -> PathPlan:
    """Plan a literal find/replace over the stored camera paths.

    Every stored path containing ``find`` has all occurrences replaced. The
    substitution is literal, not a regular expression, and the path is not
    normalised first, so a Windows-style ``--find`` such as ``D:\\photos``
    matches the stored value verbatim. Paths not containing ``find`` are left
    alone and recorded as unchanged.

    Every resulting path is checked with :func:`os.path.isfile`; failures land
    in :attr:`PathPlan.missing` for the caller's all-or-nothing gate. Nothing
    is prompted and no directory is walked.

    Args:
        paths: Camera id to currently stored filepath.
        find: Literal substring to look for.
        replace: Literal replacement.

    Returns:
        PathPlan: The planned changes.
    """
    plan = PathPlan(stored=dict(paths))
    for cam_id in sorted(paths):
        stored = paths[cam_id]
        if not stored:
            continue
        if find not in stored:
            plan.unchanged[cam_id] = stored
            continue
        new_path = stored.replace(find, replace)
        if new_path == stored:
            plan.unchanged[cam_id] = stored
            continue
        if not os.path.isfile(new_path):
            plan.missing[cam_id] = new_path
        _record(plan, cam_id, stored, new_path, path_dirname(stored))
    return plan


# ---------------------------------------------------------------------------
# Project-file (YAML) layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YamlPathKey:
    """One path-bearing key in a project YAML.

    Attributes:
        key: The YAML key as written in the file.
        is_dir: Whether the value names a directory rather than a file.
        conventions: ``{id}``-templated basenames tried, in priority order,
            when the stored value does not resolve.
    """

    key: str
    is_dir: bool
    conventions: Tuple[str, ...] = ()


# Mirrors the keys read by ProjectInitializer.init_with_yaml and the filenames
# assigned by init_with_path. Note the YAML key is ``thumbnails_path`` while the
# initializer attribute is ``thumbnail_path``; the key is what appears on disk.
YAML_PATH_KEYS: Tuple[YamlPathKey, ...] = (
    YamlPathKey("ply", False, ("{id}_dec50M.ply", "{id}.ply")),
    YamlPathKey("pcd", False, ("{id}_dec50M.ply", "{id}.ply")),  # legacy alias
    YamlPathKey("cams_xml", False, ("{id}.cams.xml",)),
    YamlPathKey("cams_meta_json", False, ("{id}.meta.json",)),
    YamlPathKey("markers", False, ("{id}_markers.csv",)),
    YamlPathKey("annotations", False, ("{id}_annotations.csv",)),
    YamlPathKey("classes", False, ("classes.csv",)),
    YamlPathKey("classifier", False, ()),
    YamlPathKey("photos_path", True, ("{id}.photos",)),
    YamlPathKey("cropped_path", True, ("{id}.cropped",)),
    YamlPathKey("thumbnails_path", True, ("{id}.thumbnails",)),
)

# Top-level ``key: value`` at zero indentation. Nested mappings (e.g. the
# entries under ``color_correction:``) are indented and therefore never match.
_YAML_KEY_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<sep>[ \t]*:[ \t]*)")

# Values needing quotes to survive a YAML round-trip.
_YAML_NEEDS_QUOTE_RE = re.compile(r"^[\s]|[\s]$|^[-?:,\[\]{}#&*!|>'\"%@`]|:\s|\s#")


@dataclass
class YamlEntry:
    """The outcome for one YAML key.

    Attributes:
        key: The YAML key.
        old: The value as written in the file.
        new: The value to write, or ``None`` when unchanged or unresolved.
        how: How it was resolved — ``unchanged``, ``project-dir``,
            ``convention``, ``found``, ``chosen``, ``skipped`` or ``missing``.
    """

    key: str
    old: Optional[str]
    new: Optional[str] = None
    how: str = "unchanged"


@dataclass
class YamlPlan:
    """Planned repairs for a project YAML, before anything is written.

    Attributes:
        yaml_path: The YAML file the plan applies to.
        project_dir: Absolute directory containing the YAML.
        project_id: Project id used to expand naming conventions.
        entries: One :class:`YamlEntry` per path key present in the file.
        resolved: Key to absolute resolved path, for keys that now resolve.
    """

    yaml_path: str
    project_dir: str
    project_id: str
    entries: List[YamlEntry] = field(default_factory=list)
    resolved: Dict[str, str] = field(default_factory=dict)

    @property
    def changes(self) -> Dict[str, str]:
        """Key to new value, for entries whose written value changes."""
        return {e.key: e.new for e in self.entries if e.new is not None}

    @property
    def unresolved(self) -> List[YamlEntry]:
        """Entries that could not be pointed at an existing file."""
        return [e for e in self.entries if e.how in ("missing", "skipped")]


def yaml_quote(value: str) -> str:
    """Quote a YAML scalar only when leaving it bare would change the parse."""
    if value == "" or _YAML_NEEDS_QUOTE_RE.search(value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def patch_yaml_paths(text: str, updates: Dict[str, str]) -> str:
    """Replace the values of top-level keys, preserving everything else.

    Rewrites only the value portion of matching ``key: value`` lines at zero
    indentation. Comments, blank lines, key order, nested mappings and any key
    not named in ``updates`` survive byte-for-byte.

    This is line-oriented rather than a PyYAML load/dump round-trip because
    PyYAML cannot preserve comments, and ``save_config_to_yaml`` already
    demonstrates the cost of rebuilding the file from scratch: it silently drops
    every key the initializer does not model.

    Args:
        text: The current contents of the YAML file.
        updates: Mapping of top-level key to the new value to write.

    Returns:
        str: The patched text. Keys not present in the file are left out; use
        :func:`patch_yaml_paths` only to repair values that already exist.
    """
    if not updates:
        return text
    out = []
    for line in text.splitlines(keepends=True):
        match = _YAML_KEY_RE.match(line)
        if match and match.group("key") in updates:
            newline = "\n" if line.endswith("\n") else ""
            value = yaml_quote(updates[match.group("key")])
            comment = _trailing_comment(line[match.end():].rstrip("\r\n"))
            out.append(
                f"{match.group('key')}{match.group('sep')}{value}{comment}{newline}"
            )
        else:
            out.append(line)
    return "".join(out)


def _trailing_comment(value_text: str) -> str:
    """Return the trailing ``# ...`` of a YAML value, including its spacing.

    A ``#`` only starts a comment when it follows whitespace and sits outside
    quotes, so a value such as ``"a # b"`` is left intact.

    Args:
        value_text: The text after the ``key:`` separator, newline stripped.

    Returns:
        str: The comment with its leading whitespace, or an empty string.
    """
    quote = None
    for i, ch in enumerate(value_text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and i > 0 and value_text[i - 1] in " \t":
            start = i
            while start > 0 and value_text[start - 1] in " \t":
                start -= 1
            return value_text[start:].rstrip()
    return ""


def relativise(resolved: str, project_dir: str) -> str:
    """Reduce a path to a bare name when it sits directly in the project dir.

    ``ProjectInitializer.__add_path_if_needed`` joins a relative YAML value onto
    the project's ``path:`` key, so a bare filename round-trips — and, unlike an
    absolute path, survives the project being moved again.

    Args:
        resolved: Absolute path to the resolved file or directory.
        project_dir: Absolute project directory.

    Returns:
        str: The basename when ``resolved`` is directly inside ``project_dir``,
        otherwise ``resolved`` unchanged.
    """
    if os.path.dirname(os.path.abspath(resolved)) == os.path.abspath(project_dir):
        return path_basename(resolved)
    return resolved


def find_by_basename(root: str, basename: str, want_dir: bool = False) -> List[str]:
    """Recursively find entries under ``root`` matching a basename.

    Args:
        root: Directory to search.
        basename: Name to match, compared case-insensitively.
        want_dir: Match directories instead of files.

    Returns:
        list[str]: Sorted absolute paths of the matches.
    """
    target = basename.lower()
    hits = []
    for dirpath, dirnames, filenames in os.walk(os.path.abspath(root)):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        names = dirnames if want_dir else filenames
        for name in names:
            if name.lower() == target:
                hits.append(os.path.join(dirpath, name))
    return sorted(hits)


def plan_yaml_repair(
    yaml_path: str,
    config: Dict[str, object],
    on_duplicate: Optional[DuplicateCallback] = None,
    on_missing: Optional[MissingCallback] = None,
) -> YamlPlan:
    """Plan repairs for every path-bearing key in a project YAML.

    Works purely on the parsed mapping and the filesystem; it never constructs a
    :class:`ProjectInitializer`, because ``initialize()`` guards the point cloud
    and cameras on truthiness alone and would crash on a dead path before any
    repair could happen.

    ``path:`` is resolved first and always set to the directory containing the
    YAML, since a stale or relative ``path:`` makes every relative value beneath
    it resolve wrongly (or depend on the caller's working directory). Each
    remaining key is then resolved against the corrected directory: an existing
    value is kept, otherwise a conventional filename is tried, then a recursive
    search by basename, then the user is asked.

    Args:
        yaml_path: Path to the project YAML.
        config: The parsed YAML mapping.
        on_duplicate: Called as ``(key, old_value, candidates)`` when a search
            finds several matches. Returns ``("pick", path)``, ``("skip", None)``
            or ``("abort", None)``.
        on_missing: Called as ``(key, old_value)`` when nothing is found.

    Returns:
        YamlPlan: The planned repairs.

    Raises:
        PathAborted: If a callback returns ``"abort"``.
        RuntimeError: If a decision is needed but the callback is ``None``.
    """
    project_dir = os.path.dirname(os.path.abspath(yaml_path))
    project_id = str(
        config.get("id") or path_basename(project_dir) or ""
    )
    plan = YamlPlan(yaml_path=yaml_path, project_dir=project_dir,
                    project_id=project_id)

    # 1. path: always names the directory the YAML actually lives in.
    old_path = config.get("path")
    old_path_str = str(old_path) if old_path is not None else None
    if old_path_str is None or os.path.abspath(old_path_str) != project_dir:
        plan.entries.append(
            YamlEntry("path", old_path_str, project_dir, "project-dir")
        )
    else:
        plan.entries.append(YamlEntry("path", old_path_str, None, "unchanged"))
    plan.resolved["path"] = project_dir

    for spec in YAML_PATH_KEYS:
        if spec.key not in config or config[spec.key] is None:
            continue
        old = str(config[spec.key])
        exists = os.path.isdir if spec.is_dir else os.path.isfile

        # 2. Resolve the stored value against the corrected project directory.
        current = old if os.path.isabs(old) else os.path.join(project_dir, old)
        if exists(current):
            plan.resolved[spec.key] = os.path.abspath(current)
            new = relativise(os.path.abspath(current), project_dir)
            if new != old:
                # Resolves fine, but shortening it to a bare name makes the
                # YAML survive the next move.
                plan.entries.append(YamlEntry(spec.key, old, new, "relative"))
            else:
                plan.entries.append(YamlEntry(spec.key, old, None, "unchanged"))
            continue

        # 3. A conventional name in the project directory.
        chosen = None
        how = ""
        for template in spec.conventions:
            candidate = os.path.join(project_dir, template.format(id=project_id))
            if exists(candidate):
                chosen, how = candidate, "convention"
                break

        # 4. Recursive search by the stored basename.
        if chosen is None:
            hits = find_by_basename(project_dir, path_basename(old), spec.is_dir)
            if len(hits) == 1:
                chosen, how = hits[0], "found"
            elif len(hits) > 1:
                if on_duplicate is None:
                    raise RuntimeError(
                        f"YAML key {spec.key!r}: {len(hits)} candidates for "
                        f"{path_basename(old)!r} and no way to ask which to use."
                    )
                action, picked = on_duplicate(spec.key, old, hits)
                if action == "abort":
                    raise PathAborted(f"Aborted while resolving {spec.key!r}.")
                if action == "skip" or not picked:
                    plan.entries.append(YamlEntry(spec.key, old, None, "skipped"))
                    continue
                chosen, how = picked, "chosen"

        # 5. Ask.
        if chosen is None:
            if on_missing is None:
                plan.entries.append(YamlEntry(spec.key, old, None, "missing"))
                continue
            action, picked = on_missing(spec.key, old)
            if action == "abort":
                raise PathAborted(f"Aborted while resolving {spec.key!r}.")
            if action == "skip" or not picked:
                plan.entries.append(YamlEntry(spec.key, old, None, "skipped"))
                continue
            chosen, how = picked, "chosen"

        chosen = os.path.abspath(chosen)
        plan.resolved[spec.key] = chosen
        new = relativise(chosen, project_dir)
        plan.entries.append(
            YamlEntry(spec.key, old, new if new != old else None, how)
        )

    return plan


def atomic_write_text(text: str, out_path: str) -> str:
    """Write text to a path atomically.

    Writes to a temporary file in the same directory and ``os.replace``s it into
    position, so an interrupted write cannot truncate the original. Mirrors
    :func:`substrata.cameras._atomic_write_json`, duplicated here rather than
    imported because this module must not depend on ``cameras``.

    Args:
        text: Contents to write.
        out_path: Destination path (created or replaced).

    Returns:
        str: The absolute path written.
    """
    out_abs = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_abs) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml.tmp", prefix=".pathrepair_",
                                    dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, out_abs)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return out_abs
