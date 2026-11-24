# Close-up photos of colonies

Divers often take close-up (macro) photos of single colonies, for identification,
next to the survey photos and a GoPro video. Metashape cannot align these close-ups:
they show a few centimetres of reef at a scale and blur the survey photos do not
share. This tutorial gives each close-up a pose in the plot, turns the point it aims
at into an annotation, and makes one page per colony to label and check.

It starts from a plot prepared as in {doc}`plot_workflow` (oriented, with its point
cloud), whose Metashape chunk also holds the GoPro video frames (aligned) and the
close-ups (added, not aligned).

| Step | Command | Writes (in the plot folder) |
| --- | --- | --- |
| 1 | `substrata camsync` (no options) | nothing: lists the sensors |
| 2 | `substrata camsync -s 2 -u 1 --match-images --dry-run` | `<id>_camsync_images.pdf` |
| 3 | `substrata camsync -s 2 -u 1 --match-images` | poses in `<id>.meta.json`, `<id>_camsync_poses.json`, backups in `backup/` |
| (4) | `substrata metashape-import-poses` | the poses in `<id>.psx` |
| 5 | `substrata cams2annos` | `<id>.annos/<id>_closeup_targets.csv` (+ sidecar) |
| 6 | `substrata explore --edit` | labels in that CSV |
| 7 | `substrata images --annotations ...` | `<id>_closeup_targets_pages.pdf` |

## Before you start

- **Photos.** The close-ups are in `<id>.closeup/` and the GoPro frames in their own
  folder (here `<id>.videoframes/`). The folder name is the camera group: `cams2annos`
  and `images` find the close-ups as group `closeup`. If the photos are elsewhere,
  move them there and run `substrata path-repair`.
- **Cameras.** Both groups are in `<id>.cams.xml` and `<id>.meta.json`
  (`metashape-export` after adding them to the chunk). The GoPro frames have poses
  (aligned by Metashape); the close-ups need none. A close-up sensor that Metashape
  never calibrated is used with its nominal lens (focal length and pixel size from
  cams.xml, no distortion).
- **Plot.** A `world_transform` (`substrata orient`) and the point cloud.
- **Software.** kornia (`pip install kornia`, or the `match` extra). The matchers'
  weights are downloaded on first use. A GPU makes a plot of 50-90 close-ups take
  minutes instead of hours.

## 1. Find the sensor ids

```bash
cd sam_sa3_40m_20250928
substrata camsync
```

Without `-s` / `-u`, in a terminal, camsync prints the sensors of `<id>.cams.xml`
(id, label, resolution, number of cameras) and asks for the pose source and the
target. Note the GoPro's id (the pose source, `-s`) and the close-up camera's id (the
updated target, `-u`), then stop with Ctrl-C. The examples below use `-s 2 -u 1`.

## 2. Dry run

```bash
substrata camsync -s 2 -u 1 --match-images --dry-run
```

What it does (`substrata.imagesync`):

1. **Time offset.** The close-up camera's and the GoPro's clocks differ by an unknown
   offset. Every close-up is matched coarsely (SIFT) with the frames at every possible
   offset, and votes for the offsets where it matches; the votes peak at the clocks'
   offset. The highest peaks are confirmed with a learned matcher (LoFTR) on up to 12
   close-ups. `--time-offset SEC` skips this search; `--offset-window MIN MAX`
   narrows it.
2. **Matching.** Each close-up is matched (LoFTR) with the aligned frames within
   `--frames` s (default 3) of its time, and if that fails with frames up to
   `--wide-frames` s off (default 8).
3. **Pose.** Each matched frame pixel is followed along the frame's ray to the cloud,
   so the close-up's pixels see known 3-D points. PnP (RANSAC, then refined) gives
   the close-up's position and orientation.
4. **Checks.** At least 8 PnP inliers, a median reprojection error below 0.7 % of the
   photo width, a range (camera to the point it aims at) within 0.5-2 × the median
   of the other close-ups, and the second matcher (DISK + LightGlue) aiming within
   2 cm of the same point across the view (`--no-check` skips this).

Nothing is saved. It prints the offset (e.g. `Using time offset -81 s`) and how many
close-ups got each status, and writes `<id>_camsync_images.pdf`.

### The statuses

| Status | Meaning | Saved as |
| --- | --- | --- |
| `posed` | Matched, and passed every check. | pose, camera enabled, `pose_status: matched` |
| `unverified` | Matched, but the two matchers aim more than 2 cm apart. LoFTR's pose is kept. | pose, camera disabled, `pose_status: unverified` |
| `estimated` | No match, or a match rejected for its error, range or a centre ray that meets no cloud. The pose is the GoPro frame's at the close-up's time + offset (or 1 s off), combined with the *rig*: where the close-up camera sits and points relative to the GoPro, the median over the posed close-ups (5 or more needed). | pose, camera disabled, `pose_status: estimated` |
| `no pose` | No match, and no estimate (no aligned frame within 1 s of its time, too few posed close-ups for the rig, or `--no-estimate`). | null pose, camera disabled |
| `no frames` | No aligned GoPro frame within `--wide-frames` s of its time, so nothing to match with. | null pose, camera disabled |

With `--no-estimate`, a close-up whose match was rejected is reported as `rejected`
and gets no pose.

On sa3_40m and sa4_40m an estimated close-up aimed 4-6 cm (median) from where its
match put it, against 21-23 cm for the GoPro frame's pose copied as it is. A diver
who moved off before or after taking the close-up makes an estimate worse.

### The check PDF

`<id>_camsync_images.pdf` shows the votes for the time offset and the rig, then
close-ups next to their GoPro frame, with the close-up's centre in the frame (red)
and where its pose aims (green). It shows every close-up that was not posed
outright, every posed one at an unusual range (beyond 1.5 × or below 1 / 1.5 × the
median), and a selection of the others (`--pdf-pairs N`).

**What good looks like.** One clear peak in the offset votes. For the posed close-ups,
the green mark on the red one: the pose aims at what the close-up shows. For an
estimated close-up the two can be centimetres apart; check that the frame shows the
same colony.

Where a close-up aims is accurate to millimetres. Its exact position is less certain:
over a flat patch, through a narrow lens, tilting the camera and shifting it sideways
look almost the same, so the camera centre can be a few cm off.

## 3. Save the poses

```bash
substrata camsync -s 2 -u 1 --match-images --time-offset -81
```

`--time-offset` reuses the offset of the dry run and saves the search. After a
confirmation (or `--yes`) it:

- backs up `<id>.meta.json` and `<id>.cams.xml` to `backup/`;
- writes the poses into `<id>.meta.json` (statuses and enabled flags as in the table;
  `pose_status` is read back when the project is loaded);
- writes `<id>_camsync_poses.json` (the same poses, for Metashape).

substrata reads camera poses from the meta JSON, so the plot now has everything it
needs. Metashape is not involved.

## 4. Optional: the poses into Metashape

```bash
substrata metashape-import-poses
```

Reads `<id>_camsync_poses.json` and gives the cameras of `<id>.psx` these poses (by
camera key, checked against the label), by running a script under Metashape's
Python. Estimated and unverified cameras get their pose too and are disabled, so you
can find and check them in Metashape (`--posed-only` leaves them alone). `--dry-run`
reports without saving.

This is only needed if you export from Metashape again: `metashape-export` rewrites
`<id>.meta.json` from the project, and the poses would otherwise be lost.

## 5. The point each close-up aims at

```bash
substrata cams2annos
```

For every camera of the group (`--group`, default `closeup`), the ray through the
photo's centre (or `--pixel X Y`) is followed from the camera, in steps of
`--search-radius` (default 3 mm), up to `--max-distance` (default 5 m). The first
cloud point within that distance of the ray is the hit; a ray through a gap is tried
again 3 × wider, up to `--max-search-radius` (default 3 cm). The hit is then refined
to the surface point nearest the ray, and its pixel in the photo is stored as the
point's view.

Writes `<id>.annos/<id>_closeup_targets.csv`, one point per close-up, with each photo
as its point's view (`cam_filepath`, `cam_x`, `cam_y`) and the extra columns
`search_radius` (the radius that hit), `target_distance` (camera to point, m) and
`target_offset_px` (the view's pixel from the centre). Points from an **estimated**
pose have `is_extrapolated` True. The sidecar has `type: camera_targets` and
`purpose: training_only`: close-ups are aimed at chosen colonies, not a random
sample, so cover summaries leave the set out.

**Check.** The command lists cameras whose ray met no cloud, or that cannot project
(no pose): they get no point. `target_distance` should be about the same for all
close-ups (around 0.3 m on the Samoa plots). Points from unverified poses are not
flagged by `is_extrapolated`: find them by their camera's `pose_status` in
`<id>.meta.json`, or in the check PDF.

In Python: `pcd.camera_targets(cams)` returns the same points as an `Annotations`
container.

## 6. Label the colonies

```bash
substrata explore --edit
```

Choose the `<id>_closeup_targets` set: each point shows its close-up. Label the
colonies as in {doc}`annotate_and_classify`.

## 7. One page per colony

```bash
substrata images --annotations sam_sa3_40m_20250928.annos/sam_sa3_40m_20250928_closeup_targets.csv
```

For a camera-targets set, `--layout auto` (the default) makes one A4 page per point:

- top left, the best overview photo, zoomed on the point (`--zoom`, at least 0.5 m
  wide, and wide enough to show 2.5 × the close-up's frame);
- top right, the close-up's full frame, with a box around the crop below;
- bottom, the close-up at 100 % (one photo pixel per dot at 300 dpi), centred on the
  point.

The point is marked in each panel, with scale bars from the camera distance, and the
header gives the plot, the annotation id, label, depth, coordinates and the CSV
columns of `--info` (default `label_conf label_source`). A close-up whose camera is
disabled (an estimated or unverified pose) is still shown, with a note to check it.
Photos are shown with a display contrast (`--contrast`, default `stretch`).

Writes `<id>_closeup_targets_pages.pdf` in the current folder (`--pdf-output` for
another path). `--labels CS_ACR` or `--ids ...` select points.

**Check.** On each page, the overview and the close-up show the same colony at the
marked point. For an estimated pose, a mismatch means the point is off: move it (in
substrata-explore) or leave it out.

Pages also work for other sets: `--closeup closeup` gives each point the close-up
camera that sees it best (points without one get an overview-only page).

## Common problems

- **"kornia" missing.** `pip install kornia` in the substrata environment.
- **No time offset confirmed.** The command stops and asks for one. Pass a known
  offset with `--time-offset`, or narrow the search with `--offset-window MIN MAX`.
  The offset of another plot filmed the same day with the same cameras is a good
  first guess.
- **Many close-ups `no frames`.** Metashape left the GoPro frames at those times
  unaligned. Align them in Metashape (Align Selected Cameras) and export again, or
  add `--pose-frames`: substrata then poses those frames itself, from matches with the
  survey photos and the aligned frames (about 10 s a frame). Their poses are used for
  this run only.
- **No GoPro frame was aligned at all.** The frames then have no lens calibration.
  `--frame-calibration OTHER.cams.xml` borrows it from another plot filmed with the
  same camera and settings (with `--pose-frames`); `--frame-calibration-orientation
  auto` (default) tells whether the camera was mounted the other way up there.
- **Close-ups turned 180°.** Handled: the offset search finds how the close-ups are
  turned in the frames (e.g. a GoPro mounted upside down).
- **Poses lost after a new Metashape export.** Run `metashape-import-poses` before
  exporting again, or run camsync again.
- **The old time-based camsync.** Without `--match-images`, camsync copies the pose of
  the GoPro frame at the same EXIF time (`--time-offset`, `--auto-offsets`). The close-up
  camera points elsewhere than the GoPro, so these poses can be metres off. Use
  `--match-images` for close-ups.
