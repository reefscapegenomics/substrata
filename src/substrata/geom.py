from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Transform:
    """Immutable 4 x 4 homogeneous transform."""

    mat: FloatArray

    # --------------------------------------------------------------------- validation

    def __post_init__(self) -> None:
        """Validate and, if needed, promote the input matrix to 4 x 4.

        Accepts:
        * (4, 4) - taken as-is
        * (3, 3) - treated as rotation, promoted to (4, 4) with bottom row/col
        * (3, 4) - rotation + translation, promoted to (4, 4)

        Raises:
            ValueError: If the supplied array cannot be interpreted as one of
                the above shapes.
        """
        arr = np.asarray(self.mat, dtype=float)

        if arr.shape == (4, 4):
            promoted = arr
        elif arr.shape == (3, 3):
            promoted = np.eye(4, dtype=float)
            promoted[:3, :3] = arr
        elif arr.shape == (3, 4):
            promoted = np.vstack((arr, [0.0, 0.0, 0.0, 1.0]))
        else:
            raise ValueError(
                "Transform must have shape (4, 4), (3, 3) or (3, 4); "
                f"got {arr.shape}."
            )

        object.__setattr__(self, "mat", promoted)

    @classmethod
    def identity(cls) -> "Transform":
        """Return the 4 x 4 identity transform."""
        return cls(np.eye(4, dtype=float))

    @classmethod
    def from_translation(cls, xyz: Tuple[float, float, float]) -> "Transform":
        """Create a pure-translation transform.

        Args:
            xyz: Offsets along (x, y, z) in the same units as the point cloud.
        """
        t = np.eye(4, dtype=float)
        t[:3, 3] = xyz
        return cls(t)

    @classmethod
    def from_scale(cls, s: float) -> "Transform":
        """Create a uniform-scale transform.

        Args:
            s: Scaling factor (non-zero).

        Raises:
            ValueError: If `s` is zero.
        """
        if s == 0:
            raise ValueError("Scale factor cannot be zero.")
        t = np.diag([s, s, s, 1.0])
        return cls(t)

    # ------------------------------------------------------------------ simple translations

    @classmethod
    def from_depth_offset(cls, dz: float) -> "Transform":
        """Translate the scene by **−dz** along Z.

        A positive ``dz`` moves geometry *toward* the viewer if +Z is up.
        """
        return cls.from_translation((0.0, 0.0, float(dz)))

    @classmethod
    def shift_to_positive_xy(cls, pcd: object) -> "Transform":
        """Build a translation that shifts a point cloud into the +X/+Y quadrant.

        Computes the offset needed so that the minimum X and Y coordinates of
        ``pcd`` become non-negative; axes already at or above zero are left
        untouched.

        Args:
            pcd: A point cloud exposing a ``points`` array of shape (N, 3).

        Returns:
            A Transform applying the computed X/Y translation offset.
        """
        min_x = np.min(pcd.points[:, 0])
        min_y = np.min(pcd.points[:, 1])
        x_offset = -min_x if min_x < 0 else 0
        y_offset = -min_y if min_y < 0 else 0
        translation_matrix = np.eye(4, dtype=float)
        translation_matrix[0, 3] = x_offset
        translation_matrix[1, 3] = y_offset

        return cls(translation_matrix)

    @classmethod
    def from_axis_angle(cls, axis: FloatArray, angle_rad: float) -> "Transform":
        """Create a rotation around an arbitrary axis.

        Args:
            axis: Length-3 vector describing the rotation axis.
            angle_rad: Rotation angle in radians.
        """
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (3,):
            raise ValueError("`axis` must be a length-3 vector.")
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        C = 1 - c
        r = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=float,
        )
        return cls(r)  # auto-promoted

    @classmethod
    def from_euler(cls, rx: float, ry: float, rz: float) -> "Transform":
        """Create a rotation from XYZ Euler angles (radians)."""
        sx, cx = np.sin(rx), np.cos(rx)
        sy, cy = np.sin(ry), np.cos(ry)
        sz, cz = np.sin(rz), np.cos(rz)
        r = np.array(
            [
                [cy * cz, -cy * sz, sy],
                [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy],
            ],
            dtype=float,
        )
        return cls(r)

    @classmethod
    def from_centroid(cls, pcd: "npt.ArrayLike | object") -> "Transform":
        """Return a translation that moves *pcd*’s centroid to the origin.

        Args:
            pcd:  • Open3D/Decorated point cloud with a ``points`` attribute, or
                  • Any `(N, 3)` array-like of XYZ coordinates.

        Returns:
            Transform: 4 × 4 matrix that translates by **−centroid**.
        """
        # Get the raw (N, 3) array of points
        points = np.asarray(pcd.points) if hasattr(pcd, "points") else np.asarray(pcd)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("`pcd` must provide an (N, 3) point array.")

        centroid = points.mean(axis=0)
        t = np.eye(4, dtype=float)
        t[:3, 3] = -centroid
        return cls(t)

    @classmethod
    def from_xy_origin(cls, xy: "npt.ArrayLike | Tuple[float, float]") -> "Transform":
        """Create a transform that shifts the X-Y origin to *xy*.

        Args:
            xy: Two-element sequence (x, y) or any array-like convertible to that.

        Returns:
            Transform: 4 × 4 matrix translating by (−x, −y, 0).

        """
        x, y = np.asarray(xy, dtype=float).flatten()[:2]
        t = np.eye(4, dtype=float)
        t[0, 3] = -x
        t[1, 3] = -y
        return cls(t)

    # ------------------------------------------------------------------ up-vector helper

    @classmethod
    def from_up_vector_legacy(cls, up: "Vector | npt.ArrayLike") -> "Transform":
        """Build a rotation that mimics the legacy get_up_vector_transform (intrinsic Z–Y–X)."""
        # Use the same angle definitions as the legacy helper:
        # theta_xz = atan2(-vx, -vz), psi_yz = atan2(-vy, -vz)
        v = up.xyz if isinstance(up, Vector) else Vector(up).xyz
        theta_xz = np.arctan2(-v[0], -v[2])
        psi_yz = np.arctan2(-v[1], -v[2])

        cb, sb = np.cos(theta_xz), np.sin(theta_xz)
        cg, sg = np.cos(psi_yz), np.sin(psi_yz)

        # Intrinsic Z–Y–X with α=0 reduces to this 3×3 (matches legacy formula)
        R = np.array(
            [
                [cb, sb * sg, sb * cg],
                [0.0, cg, -sg],
                [-sb, cb * sg, cb * cg],
            ],
            dtype=float,
        )

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        return cls(T)

    @classmethod
    def from_up_vector(cls, up: "Vector | npt.ArrayLike") -> "Transform":
        """Rotate so the supplied up-vector maps to global +Z."""
        v = up.xyz if isinstance(up, Vector) else Vector(up).xyz
        v = v / np.linalg.norm(v)

        z = np.array([0.0, 0.0, 1.0])

        # Collinear cases
        if np.allclose(v, z):
            return cls.identity()
        if np.allclose(v, -z):
            return cls.from_axis_angle([1.0, 0.0, 0.0], np.pi)

        # Find rotation that sends v → z
        axis = np.cross(v, z)
        s = np.linalg.norm(axis)
        c = float(np.dot(v, z))
        axis = axis / s
        angle = np.arctan2(s, c)

        return cls.from_axis_angle(axis, angle)

    @classmethod
    def align_x_to_vector(cls, v: "Vector | npt.ArrayLike") -> "Transform":
        """Rotate so the global +X axis becomes parallel to *v* (in XY-plane)."""
        vec = v.xyz if isinstance(v, Vector) else Vector(v).xyz
        vec[2] = 0  # force into XY-plane
        if np.allclose(vec[:2], 0):
            return cls.identity()  # already along X or vector is zero

        # angle between +X (1,0) and projected vector, measured CCW about +Z
        theta = np.arctan2(vec[1], vec[0])
        # legacy code used a clockwise (negative) rotation;
        # keep that so the visual result stays identical
        return cls.from_euler(rx=0.0, ry=0.0, rz=-theta)  # rotate about Z

    # ------------------------------------------------------------------ upslope helper
    @classmethod
    def ensure_pos_y_is_upslope(cls, pcd: "npt.ArrayLike | object") -> "Transform":
        """
        Rotate 180 ° about the global +Z axis so the cloud’s **+Y axis points
        “upslope”** (positive correlation with Z).

        It reproduces the behaviour of the legacy
        ``get_y_axis_facing_upslope_transform``:

            • If corr(Y, Z) < 0  → flip by π around Z.
            • Otherwise          → identity.

        Parameters
        ----------
        pcd
            Object exposing a ``points`` attribute (Open3D / decorated cloud) or
            an ``(N, 3)`` array-like of XYZ coordinates.

        Returns
        -------
        Transform
            The appropriate rotation (identity or 180 ° Z-rotation).
        """
        # Obtain the raw (N, 3) point array
        pts = np.asarray(pcd.points) if hasattr(pcd, "points") else np.asarray(pcd)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("`pcd` must provide an (N, ≥3) point array.")

        if len(pts) < 2:  # not enough data for correlation
            return cls.identity()

        r = np.corrcoef(pts[:, 1], pts[:, 2])[0, 1]  # corr(Y, Z)
        # after the clockwise X-axis alignment, positive r means Y points
        # upslope, so we need to rotate when r < 0
        return cls.from_euler(0.0, 0.0, np.pi) if r < 0 else cls.identity()

    @classmethod
    def align_y_upslope(
        cls,
        pcd: "npt.ArrayLike | object",
        max_points: int = 1_000_000,
        snap_to_plot_axes: bool = True,
        min_elongation: float = 2.0,
    ) -> "Transform":
        """Rotate about +Z so +Y points upslope, square to the plot's footprint.

        Fits a plane ``z = a*x + b*y + c`` (least squares, on a random subset
        of at most ``max_points`` points) whose gradient ``(a, b)`` is the
        steepest upslope direction (the fall line). Unlike align_x_to_vector +
        ensure_pos_y_is_upslope, this works whichever way the plot is
        elongated, so the shallow end always ends up at +Y.

        Survey lines rarely run exactly along (or across) the fall line, so
        with ``snap_to_plot_axes`` the rotation instead uses whichever of the
        footprint's principal XY axes (long or short) is closest to the fall
        line, signed to point upslope. The plot's edges then line up with X/Y
        (axis-aligned bounding boxes fit snugly) while +Y still points
        (approximately) upslope. Footprints that are not clearly elongated
        (variance ratio below ``min_elongation``) use the fall line as-is.

        Args:
            pcd: Object exposing ``points`` or an ``(N, 3)`` array-like.
            max_points: Cap on points used for the fits.
            snap_to_plot_axes: Align to the footprint's principal axes rather
                than the exact fall line (see above).
            min_elongation: Minimum ratio of the principal XY variances for
                the footprint axes to be used.

        Returns:
            The Z-rotation (identity if the cloud is flat).
        """
        pts = np.asarray(pcd.points) if hasattr(pcd, "points") else np.asarray(pcd)
        if len(pts) > max_points:
            rng = np.random.default_rng(0)
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        if len(pts) < 3:
            return cls.identity()
        design = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        (a, b, _), *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
        if np.hypot(a, b) < 1e-12:
            return cls.identity()
        direction = np.array([a, b])

        if snap_to_plot_axes:
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            variances, axes = np.linalg.eigh(np.cov(xy.T))
            if variances[0] > 0 and variances[1] / variances[0] >= min_elongation:
                # Footprint axis closest to the fall line, pointing upslope
                axis = max(axes.T, key=lambda v: abs(v @ direction))
                direction = axis * np.sign(axis @ direction)

        return cls.from_euler(
            0.0, 0.0, np.pi / 2 - np.arctan2(direction[1], direction[0])
        )

    @classmethod
    def align_normal_to_z(cls, normal: "Vector | npt.ArrayLike") -> "Transform":
        """Rotate so *normal* becomes the global +Z axis.

        Args:
            normal: Plane normal to be aligned.

        Returns:
            Transform: 4 × 4 rotation matrix.
        """
        n = normal.xyz if isinstance(normal, Vector) else Vector(normal).xyz
        n = n / np.linalg.norm(n)

        # Find any vector not parallel to n
        ref = (
            np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )

        x_n = np.cross(ref, n)
        x_n /= np.linalg.norm(x_n)
        y_n = np.cross(n, x_n)

        R = np.vstack((x_n, y_n, n))  # rows → new basis
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        return cls(T)

    @classmethod
    def from_point_correspondences(
        cls,
        src: npt.ArrayLike,
        dst: npt.ArrayLike,
        with_scaling: bool = True,
    ) -> "Transform":
        """Least-squares transform mapping ``src`` points onto ``dst`` points.

        Closed-form Umeyama (1991) solution: a similarity transform (uniform
        scale, rotation, translation) or, with ``with_scaling=False``, a rigid
        one. Reflections are excluded.

        Args:
            src: (N, 3) source points.
            dst: (N, 3) corresponding destination points.
            with_scaling: If True, also estimate a uniform scale.

        Returns:
            Transform: 4 × 4 matrix with ``dst ≈ T.apply_to_points(src)``.

        Raises:
            ValueError: If the shapes differ, fewer than 3 points are given, or
                the source points are collinear.
        """
        p = np.asarray(src, dtype=float)
        q = np.asarray(dst, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or p.shape != q.shape:
            raise ValueError("`src` and `dst` must both have shape (N, 3).")
        if len(p) < 3:
            raise ValueError(f"At least 3 point pairs are required; got {len(p)}.")

        p_mean, q_mean = p.mean(axis=0), q.mean(axis=0)
        p_c, q_c = p - p_mean, q - q_mean
        sv = np.linalg.svd(p_c, compute_uv=False)
        if sv[0] == 0 or sv[1] <= 1e-9 * sv[0]:
            raise ValueError("Source points are degenerate (coincident or collinear).")

        # Cross-covariance and its SVD; D flips the last axis to avoid reflections
        U, S, Vt = np.linalg.svd(q_c.T @ p_c / len(p))
        D = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            D[2, 2] = -1.0
        R = U @ D @ Vt
        s = 1.0
        if with_scaling:
            s = float(np.trace(np.diag(S) @ D) / (p_c**2).sum(axis=1).mean())

        T = np.eye(4, dtype=float)
        T[:3, :3] = s * R
        T[:3, 3] = q_mean - s * R @ p_mean
        return cls(T)

    # --------------------------------------------------------------------- numeric operators

    def __matmul__(self, other: "Transform") -> "Transform":
        """Compose two transforms (`self` after `other`)."""
        return Transform(other.mat @ self.mat)

    def __rmatmul__(self, other: "Transform") -> "Transform":
        """Compose two transforms (`other` after `self`)."""
        return Transform(self.mat @ other.mat)

    # --------------------------------------------------------------------- convenience operations

    def inverse(self) -> "Transform":
        """Return the inverse transform."""
        return Transform(np.linalg.inv(self.mat))

    def apply_to_points(self, points: FloatArray) -> FloatArray:
        """Apply the transform to an (N, 3) array of XYZ points."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("`points` must have shape (N, 3).")
        homog = np.c_[pts, np.ones(len(pts))]
        return (homog @ self.mat.T)[:, :3]

    # --------------------------------------------------------------------- interoperability helpers

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        """Return the underlying 4 × 4 matrix for NumPy / Open3D.

        NumPy calls this automatically inside ``np.asarray(T)``.  Because
        Open3D’s ``geometry*.transform`` functions do exactly that, you can
        now write::

            pcd.transform(T)          # instead of pcd.transform(T.mat)

        Args:
            dtype: Optional dtype to cast to (NumPy passes this through).

        Returns:
            numpy.ndarray: The (4, 4) homogeneous matrix.
        """
        return self.mat.astype(dtype) if dtype is not None else self.mat

    def to_open3d(self) -> np.ndarray:
        """Return a view of the matrix suitable for Open3D."""
        return self.mat

    def as_ndarray(self) -> FloatArray:
        """Return a **copy** of the underlying 4 x 4 array."""
        return self.mat.copy()


@dataclass(frozen=True, slots=True)
class Vector:
    """Immutable homogeneous 4-vector.

    Accepts a 3- or 4-element vector. If a 3-vector is provided it is promoted
    to homogeneous form by appending ``1.0``.
    """

    vec: FloatArray

    # ------------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        """Validate and promote to 4 elements when necessary.

        Raises:
            ValueError: If the supplied array is not length 3 or 4.
        """
        arr = np.asarray(self.vec, dtype=float).flatten()
        if arr.shape == (3,):
            arr = np.append(arr, 1.0)
        elif arr.shape != (4,):
            raise ValueError("Vector must have length 3 or 4; got " f"{arr.shape[0]}.")

        object.__setattr__(self, "vec", arr)

    # ------------------------------------------------------------------ convenience properties
    @property
    def xyz(self) -> FloatArray:
        """Return the Cartesian (x, y, z) part."""
        return self.vec[:3]

    # ------------------------------------------------------------------ numeric helpers
    def dot(self, other: "Vector | npt.ArrayLike") -> float:
        """Return the dot product with another vector (auto-promotes)."""
        o = other if isinstance(other, Vector) else Vector(other)
        return float(np.dot(self.vec, o.vec))

    # Use the @ operator for dot products
    def __matmul__(self, other: "Vector | npt.ArrayLike") -> float:  # self @ other
        return self.dot(other)

    def __rmatmul__(self, other: "Vector | npt.ArrayLike") -> float:  # other @ self
        return Vector(other).dot(self)

    # ------------------------------------------------------------------ interoperability
    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        """Return the underlying 4-element array for NumPy / Open3D."""
        return self.vec.astype(dtype) if dtype is not None else self.vec

    def as_ndarray(self) -> FloatArray:
        """Return a copy of the underlying 4-vector."""
        return self.vec.copy()

    # ------------------------------------------------------------------ orientation helpers
    def intrinsic_zyx_angles(self) -> tuple[float, float, float]:
        """Return the intrinsic Z–Y–X (yaw, pitch, roll) angles in **radians**.

        The angles describe the rotation that aligns the global +Z axis with this
        vector’s direction using the Tait-Bryan (yaw–pitch–roll) convention:

        1. **γ**  (roll)  – rotation about **X** (default 0 because a single
           direction doesn’t constrain roll).
        2. **β**  (pitch) – rotation about **Y** to tilt in the XZ-plane so the
           vector’s X/Z ratio matches.
        3. **α**  (yaw)   – rotation about **Z**; for a single vector we set it
           to 0 because yaw is also unconstrained.

        Returns:
            tuple: (α, β, γ) = (yaw_Z, pitch_Y, roll_X) in radians.
        """
        x, y, z = self.xyz / np.linalg.norm(self.xyz)

        # Pitch (β): angle between vector and +Z in the XZ-plane (Y-axis rotation)
        beta = np.arctan2(x, z)  # +Y rotation: left/right tilt

        # Roll (γ): angle between vector and its XZ projection in the YZ-plane
        gamma = -np.arctan2(y, np.hypot(x, z))  # +X rotation: up/down tilt

        alpha = 0.0  # yaw about Z is undefined for a lone direction; set to 0

        return alpha, beta, gamma


def calculate_vector_angle(opposite_y, adjacent_x):
    """Return the angle (in degrees) between a 2D vector and the x-axis.

    Args:
        opposite_y: The vertical (y) component of the vector.
        adjacent_x: The horizontal (x) component of the vector.

    Returns:
        Angle in degrees between the vector and the x-axis.

    Formerly:
        calculate_angle_between_two_lengths
    """
    return np.degrees(np.arctan2(opposite_y, adjacent_x))


def fit_correspondences_rejecting_outliers(
    src: npt.ArrayLike,
    dst: npt.ArrayLike,
    with_scaling: bool = True,
    max_residual: float | None = None,
    mad_k: float = 5.0,
    max_iter: int = 20,
) -> Tuple[Transform, np.ndarray]:
    """Fit ``src`` → ``dst`` while iteratively discarding outlier pairs.

    Repeatedly fits :meth:`Transform.from_point_correspondences` on the current
    inliers and recomputes the residuals of *all* pairs, keeping those within
    the threshold, until the inlier set no longer changes. Pairs dropped in an
    early (outlier-biased) fit can therefore return once the fit improves.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) corresponding destination points.
        with_scaling: If True, also estimate a uniform scale.
        max_residual: Fixed residual threshold in ``dst`` units. If None, a
            robust threshold of ``median + mad_k * 1.4826 * MAD`` of the
            current inlier residuals is used.
        mad_k: Number of (MAD-based) standard deviations for the automatic
            threshold.
        max_iter: Maximum number of refits.

    Returns:
        (transform, inliers): the final transform and a boolean mask of the
            pairs used for it.

    Raises:
        ValueError: If the initial fit fails (see
            :meth:`Transform.from_point_correspondences`).
    """
    p = np.asarray(src, dtype=float)
    q = np.asarray(dst, dtype=float)
    inliers = np.ones(len(p), dtype=bool)
    T = Transform.from_point_correspondences(p, q, with_scaling=with_scaling)
    for _ in range(max_iter):
        residuals = np.linalg.norm(T.apply_to_points(p) - q, axis=1)
        if max_residual is not None:
            threshold = max_residual
        else:
            r_in = residuals[inliers]
            median = np.median(r_in)
            threshold = median + mad_k * 1.4826 * np.median(np.abs(r_in - median))
        new_inliers = residuals <= threshold
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            break
        try:
            T_new = Transform.from_point_correspondences(
                p[new_inliers], q[new_inliers], with_scaling=with_scaling
            )
        except ValueError:
            break
        T, inliers = T_new, new_inliers
    return T, inliers


def transform_coords(coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transforms a 3D coordinate using a 4x4 homogeneous transformation matrix.

    Args:
        coords (np.ndarray): A 3-element vector representing the 3D point.
        transform (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        np.ndarray: The transformed 3D coordinate.

    Raises:
        ValueError: If `coords` is not a 3-element vector or `transform` is not a 4x4 matrix.
    """
    # Accept our Transform class or anything with .mat/.matrix
    if hasattr(transform, "mat"):
        transform = transform.mat
    elif hasattr(transform, "matrix"):
        transform = transform.matrix

    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix must be of shape (4,4).")

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3,):
        raise ValueError("Coordinate must be a 3-element vector.")

    hom_coords = np.append(coords, 1)
    transformed = np.dot(transform, hom_coords)[:3]
    return transformed


def offset_in_camera_frame(coords, camera_transform, offset) -> np.ndarray:
    """Move ``coords`` by an offset expressed in a camera's own frame.

    Used for devices mounted on a camera rig (e.g. a sonar or depth sensor):
    the offset follows the camera's orientation in every photo.

    Args:
        coords: Camera centre ``(x, y, z)``.
        camera_transform: 4x4 camera-to-world transform (row-major, flat or
            nested). Its rotation columns are normalised, so a transform that
            includes a uniform scale is handled.
        offset: ``(x, y, z)`` in the camera frame (x: image right, y: image
            down, z: view direction), in the same units as ``coords``.

    Returns:
        np.ndarray: The offset position ``coords + R @ offset``.
    """
    rot = np.array(camera_transform, dtype=float).reshape((4, 4))[:3, :3]
    rot = rot / np.linalg.norm(rot, axis=0)
    return np.asarray(coords, dtype=float) + rot @ np.asarray(offset, dtype=float)


def sample_points_along_line(start_coord, target_coord, step_size):
    """
    Sample points along a line segment between start_coord and target_coord
    author: PB
    """
    vector = target_coord - start_coord
    dist = np.linalg.norm(vector)
    if dist < step_size:
        return np.array([start_coord])
    unit_vector = vector / dist
    n = int(dist // step_size)
    dists = np.arange(0, n * step_size, step_size)
    if dists.size == 0 or dists[-1] < dist:
        dists = np.append(dists, dist)
    return start_coord + dists[:, None] * unit_vector


def slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between unit vectors u and v
    at fraction t in [0,1].
    """
    dot_uv = np.dot(u, v)
    dot_uv = np.clip(dot_uv, -1.0, 1.0)
    angle = np.arccos(dot_uv)

    # If angle ~ 0, vectors are almost identical → no arc.
    if angle < 1e-8:
        return u
    # If angle ~ pi, vectors are nearly opposite → fallback to linear
    # to avoid singularities.
    elif abs(angle - np.pi) < 1e-8:
        return (1.0 - t) * u + t * v

    # Normal slerp
    return (np.sin((1.0 - t) * angle) / np.sin(angle)) * u + (
        np.sin(t * angle) / np.sin(angle)
    ) * v


# ----------------------------------------------------------------------- geodesy

# WGS84 ellipsoid (semi-major axis in metres, flattening)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def ecef_to_geodetic(xyz: "npt.ArrayLike") -> Tuple[float, float, float]:
    """Convert a WGS84 geocentric (ECEF) point to geodetic coordinates.

    Uses Bowring's closed-form approximation (sub-millimetre for points near
    the Earth's surface).

    Args:
        xyz: ECEF ``(x, y, z)`` in metres.

    Returns:
        Tuple of ``(lat_deg, lon_deg, height_m)`` (ellipsoidal height).
    """
    x, y, z = np.asarray(xyz, dtype=float)
    a = WGS84_A
    b = a * (1.0 - WGS84_F)
    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = (a * a - b * b) / (b * b)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3
    )
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float = 0.0) -> FloatArray:
    """Convert WGS84 geodetic coordinates to a geocentric (ECEF) point.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        h: Ellipsoidal height in metres.

    Returns:
        ECEF ``(x, y, z)`` in metres.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + h) * np.sin(lat),
        ]
    )


def ecef_to_enu_transform(lat_deg: float, lon_deg: float, h: float = 0.0) -> FloatArray:
    """Build the 4x4 transform from ECEF to a local East-North-Up frame.

    The ENU origin is the point ``(lat_deg, lon_deg, h)``; +X points east, +Y
    north and +Z up (along the ellipsoid normal).

    Args:
        lat_deg: Origin latitude in degrees.
        lon_deg: Origin longitude in degrees.
        h: Origin ellipsoidal height in metres.

    Returns:
        4x4 homogeneous transform (metres in, metres out).
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    rot = np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )
    t = np.eye(4, dtype=float)
    t[:3, :3] = rot
    t[:3, 3] = -rot @ geodetic_to_ecef(lat_deg, lon_deg, h)
    return t


def chunk_transform_to_enu(
    chunk_transform: "npt.ArrayLike", ref_point: "npt.ArrayLike"
) -> Tuple[FloatArray, float, Tuple[float, float]]:
    """Convert a Metashape chunk transform into a local metric ENU transform.

    Metashape's ``chunk.transform.matrix`` maps chunk-internal coordinates to
    geocentric (ECEF) metres. Applying it directly yields Earth-centred
    coordinates (~6e6 m, Z along the Earth's axis), so it is composed with an
    ECEF -> East-North-Up rotation at the geodetic position of ``ref_point``.
    The ENU origin is placed at ellipsoidal height 0, so the resulting Z equals
    the ellipsoidal height (i.e. -depth when camera references are depths).

    Args:
        chunk_transform: 4x4 chunk-to-ECEF transform (similarity transform).
        ref_point: A point in chunk coordinates (e.g. the cloud centre) whose
            latitude/longitude defines the ENU origin.

    Returns:
        Tuple of ``(transform, scale_factor, (lat_deg, lon_deg))``: the 4x4
        chunk-to-ENU transform, the uniform scale it contains (metres per
        chunk unit) and the ENU origin.
    """
    chunk = np.asarray(chunk_transform, dtype=float)
    if chunk.shape != (4, 4):
        raise ValueError(f"chunk_transform must have shape (4, 4); got {chunk.shape}")
    scale = float(np.cbrt(np.linalg.det(chunk[:3, :3])))
    if scale <= 0:
        raise ValueError("chunk_transform has a non-positive determinant")
    ref_ecef = transform_coords(np.asarray(ref_point, dtype=float), chunk)
    lat, lon, _ = ecef_to_geodetic(ref_ecef)
    enu = ecef_to_enu_transform(lat, lon, 0.0)
    return enu @ chunk, scale, (lat, lon)
