"""Square grounding: detect the board and map square names to image regions.

Chosen approach (see ``docs/architecture.md``): a learned **YOLO-nano 4-corner
detector** on the overhead camera locates the board corners, then a deterministic
**homography** grounds all 64 squares. This module implements the deterministic
geometry (:func:`grid_from_corners`) and the interfaces (:class:`CornerDetector`,
:class:`SquareGrounder`). The learned detector is a ``CornerDetector`` trained on
the cloud GPU and exported to ONNX/OpenVINO — the 1b.2 follow-up.
:class:`FixedCornerDetector` supplies corners by hand as the calibration bootstrap.

Geometry conventions match ``board_state``: canonical board corners are
a1=(0,0), h1=(N,0), h8=(N,N), a8=(0,N) where N=8 (an 8x8 board — NxN boards are
out of scope, chess is 8x8). The homography is scale-/perspective-invariant, so
the *physical* size of the board does not matter. Grounding produces both exact
perspective quads (:class:`SquareQuad`) and their AABBs (:class:`ImageRegion`).

**Orientation:** which physical corner is a1 is a calibration responsibility.
``BoardCorners`` must be given in a1, h1, h8, a8 order; it validates that they
form a non-degenerate convex quad (catching swapped/garbage corners), and
:meth:`BoardCorners.rotated` relabels for a board placed at a known rotation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion, Point, SquareQuad
from chess_robot.chess.board_state import FILES, NUM_FILES, RANKS, Square

# A raw camera frame (e.g. a numpy HxWxC array); typed loosely.
Image = object


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _is_convex_quad(points: list[Point]) -> bool:
    """True iff the 4 ordered points form a simple, non-degenerate convex quad."""
    signs: list[bool] = []
    n = len(points)
    for i in range(n):
        cross = _cross(points[i], points[(i + 1) % n], points[(i + 2) % n])
        if cross == 0:  # collinear corner
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


@dataclass(frozen=True)
class BoardCorners:
    """Image-pixel coordinates of the four board corners (a1, h1, h8, a8 order)."""

    a1: Point
    h1: Point
    h8: Point
    a8: Point

    def __post_init__(self) -> None:
        points = [self.a1, self.h1, self.h8, self.a8]
        if len({tuple(p) for p in points}) != 4:
            raise ValueError(f"board corners must be 4 distinct points, got {points}")
        if not _is_convex_quad(points):
            raise ValueError(
                "board corners must form a convex quad in a1, h1, h8, a8 order "
                f"(check the ordering / which corner is a1): {points}"
            )

    def rotated(self, quarter_turns: int = 1) -> BoardCorners:
        """Relabel corners for a board placed rotated by ``quarter_turns`` * 90 deg.

        Each quarter turn cycles the corner labels by one position around the
        board; ``rotated(4)`` returns the original labelling.
        """
        ordered = [self.a1, self.h1, self.h8, self.a8]
        k = quarter_turns % 4
        rolled = ordered[-k:] + ordered[:-k] if k else ordered
        return BoardCorners(a1=rolled[0], h1=rolled[1], h8=rolled[2], a8=rolled[3])


def _solve_homography(src: list[Point], dst: list[Point]) -> np.ndarray:
    """Solve the 3x3 homography mapping the 4 ``src`` points to ``dst`` (h33=1)."""
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v])
        rhs.append(u)
        rhs.append(v)
    h = np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))
    return np.array(
        [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]], dtype=float
    )


def _project(homography: np.ndarray, x: float, y: float) -> Point:
    denom = homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    u = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denom
    v = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denom
    return float(u), float(v)


def grid_from_corners(corners: BoardCorners) -> GroundedGrid:
    """Ground all 64 squares via a homography from the 4 board corners.

    Produces both the exact perspective quad per square and its AABB region.
    """
    size = float(NUM_FILES)
    src = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    dst = [corners.a1, corners.h1, corners.h8, corners.a8]
    homography = _solve_homography(src, dst)

    regions: dict[Square, ImageRegion] = {}
    quads: dict[Square, SquareQuad] = {}
    for file_index, file in enumerate(FILES):
        for rank in RANKS:
            r = rank - 1
            quad = SquareQuad(
                corners=(
                    _project(homography, file_index, r),
                    _project(homography, file_index + 1, r),
                    _project(homography, file_index + 1, r + 1),
                    _project(homography, file_index, r + 1),
                )
            )
            square = Square(file, rank)
            quads[square] = quad
            regions[square] = quad.bounding_region()
    return GroundedGrid(regions=regions, quads=quads)


class CornerDetector(ABC):
    """Locate the four board corners in an image.

    The learned YOLO-nano detector (trained on the cloud GPU, ONNX/OpenVINO on
    the laptop) is the target implementation — the 1b.2 follow-up.
    """

    @abstractmethod
    def detect(self, image: Image) -> BoardCorners:
        raise NotImplementedError


class SquareGrounder(ABC):
    """Map an image to a grounded grid (square name -> region/quad)."""

    @abstractmethod
    def ground(self, image: Image) -> GroundedGrid:
        raise NotImplementedError


class FixedCornerDetector(CornerDetector):
    """Calibration bootstrap: return hand-supplied corners, ignoring the image."""

    def __init__(self, corners: BoardCorners) -> None:
        self._corners = corners

    def detect(self, image: Image = None) -> BoardCorners:
        return self._corners


class CornerSquareGrounder(SquareGrounder):
    """Ground squares by detecting corners then applying the homography."""

    def __init__(self, detector: CornerDetector) -> None:
        self._detector = detector

    def ground(self, image: Image = None) -> GroundedGrid:
        return grid_from_corners(self._detector.detect(image))
