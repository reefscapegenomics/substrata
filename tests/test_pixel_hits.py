"""cameras.pixel_hits: the cameras that see a point, all at once, as Camera.get_pixel_coords one at a time."""

import numpy as np

from substrata.cameras import Camera, Sensor, _HITS_CACHE, pixel_hits


def _cameras(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sensors = [Sensor(i, f"s{i}", {"width": w, "height": h},
                      {"f": f, "cx": 3.0, "cy": -2.0, "k1": -0.1, "k2": 0.02, "k3": 0.0, "p1": 1e-4, "p2": -2e-4})
               for i, (w, h, f) in enumerate([(4000, 3000, 3500.0), (2848, 2816, 3100.0)])]
    cams = []
    for i in range(n):  # looking down (camera z towards -z) from 1-4 m over a 20 m square, tilted a little
        a, b = rng.normal(0, 0.3, 2)
        Rx = np.array([[1, 0, 0], [0, np.cos(np.pi + a), -np.sin(np.pi + a)], [0, np.sin(np.pi + a), np.cos(np.pi + a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        T = np.eye(4)
        T[:3, :3] = Ry @ Rx
        T[:3, 3] = [*rng.uniform(0, 20, 2), rng.uniform(1, 4)]
        c = Camera(cam_id=i, camera_transform=T, coords=T[:3, 3].copy(), path=f"/photos/{i}.jpg")
        c.sensor = sensors[i % 2]
        c.enabled = i % 17 != 0
        cams.append(c)
    return cams


def _one_at_a_time(cams, p, enabled_only=True):
    out = []
    for c in cams:
        if enabled_only and c.enabled is False:
            continue
        x, y, d, rel = c.get_pixel_coords(p, use_orig_coords=True)
        if x is not None:
            out.append((c, x, y, d, rel))
    return out


def test_same_as_one_camera_at_a_time():
    _HITS_CACHE.clear()
    cams = _cameras()
    rng = np.random.default_rng(1)
    seen = 0
    for p in np.c_[rng.uniform(0, 20, (40, 2)), rng.uniform(-0.5, 0.5, 40)]:
        for enabled_only in (True, False):
            old, new = _one_at_a_time(cams, p, enabled_only), pixel_hits(cams, p, enabled_only=enabled_only)
            assert [a[0] for a in old] == [b[0] for b in new]
            assert [a[1:3] for a in old] == [b[1:3] for b in new]
            assert np.allclose([a[3:] for a in old], [b[3:] for b in new]) if old else not new
            seen += len(old)
    assert seen > 100  # (the points are seen: a real comparison)


def test_without_sensor_one_at_a_time():
    _HITS_CACHE.clear()
    cams = _cameras(20)
    p = np.array([10.0, 10.0, 0.0])
    want = _one_at_a_time(cams, p)
    cams.append(Camera(cam_id=99, camera_transform=np.eye(4), coords=np.zeros(3), path="/photos/99.jpg"))  # (no sensor: never in view)
    cams[-1].get_pixel_coords = lambda *a, **k: (None, None, None, None)
    assert [b[0] for b in pixel_hits(cams, p)] == [a[0] for a in want]


def test_enabled_read_afresh_and_pose_changes_restack():
    _HITS_CACHE.clear()
    cams = _cameras()
    p = np.array([10.0, 10.0, 0.0])
    hit = pixel_hits(cams, p)[0][0]
    hit.enabled = False  # (after the cameras were stacked: not stale)
    assert hit not in [h[0] for h in pixel_hits(cams, p)]
    assert [h[0] for h in pixel_hits(cams, p)] == [a[0] for a in _one_at_a_time(cams, p)]
    hit.enabled = True
    T = np.array(hit.camera_transform, float)
    T[:3, 3] += [50.0, 0, 0]  # moved away: no longer sees the point
    hit.camera_transform, hit.coords = T, T[:3, 3].copy()
    hit.reverse_transform_coords(np.eye(4))  # (its orig pose changed in place: the cache is cleared)
    assert hit not in [h[0] for h in pixel_hits(cams, p)]
    assert [h[0] for h in pixel_hits(cams, p)] == [a[0] for a in _one_at_a_time(cams, p)]


def test_a_point_behind_the_camera_is_not_seen():
    _HITS_CACHE.clear()
    cams = _cameras(30)
    c = cams[1]
    above = np.asarray(c.coords, float) - 5 * np.asarray(c.camera_transform, float)[:3, 2]  # (behind it)
    assert c.get_pixel_coords(above, use_orig_coords=True)[0] is None
    assert not c.project_points(above, use_orig_coords=True)[-1]
    assert c not in [h[0] for h in pixel_hits(cams, above)]
