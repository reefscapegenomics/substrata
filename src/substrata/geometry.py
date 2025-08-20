from __future__ import annotations

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
        return cls.from_translation((0.0, 0.0, -float(dz)))

    @classmethod
    def shift_to_positive_xy(cls, pcd: object) -> "Transform":
        min_x = np.min(pcd.points[:, 0])
        min_y = np.min(pcd.points[:, 1])
        x_offset = -min_x if min_x < 0 else 0
        y_offset = -min_y if min_y < 0 else 0
        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = x_offset
        translation_matrix[1, 3] = y_offset

        return translation_matrix

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
    if hasattr(transform, "__class__") and transform.__class__.__name__ == "Transform":
        transform = transform.matrix
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix must be of shape (4,4).")
    if coords.shape != (3,):
        raise ValueError("Coordinate must be a 3-element vector.")
    hom_coords = np.append(coords, 1)
    transformed = np.dot(transform, hom_coords)[:3]
    return transformed


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
