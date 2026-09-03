from dataclasses import dataclass, field

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from eyepop.depth_map import DepthMap
from eyepop.point_cloud import PointCloud


class EyePopPlot:
    def __init__(self, axes: Axes):
        self.axes = axes

    def prediction(self, prediction: dict):
        if prediction is None:
            return
        self.depth(prediction)
        objects = prediction.get('objects', None)
        if objects is not None:
            for obj in objects:
                self.object(obj)

    def depth(self, prediction: dict, opacity: float = 0.5):
        """Overlay the prediction's frame-level depth map as a turbo heatmap.

        Near = warm (red/yellow), far = cool (blue), sky (+inf) transparent.
        """
        depth_map = DepthMap.from_prediction(prediction)
        if depth_map is None:
            return
        values = depth_map.array
        sky = np.isinf(values)
        finite = values[~sky]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            span = (hi - lo) or 1.0
            normalized = np.clip((np.where(sky, hi, values) - lo) / span, 0.0, 1.0)
        else:
            normalized = np.zeros_like(values)
        # near = warm end of turbo, far = cool end; sky masked -> transparent
        colored = plt.get_cmap('turbo')(1.0 - normalized)
        colored[..., 3] = np.where(sky, 0.0, opacity)
        extent = (0, prediction.get('source_width', depth_map.width),
                  prediction.get('source_height', depth_map.height), 0)
        self.axes.imshow(colored, extent=extent, interpolation='nearest')

    def object(self, obj: dict):
        label = self._label(obj)
        min_dim = min(obj['height'], obj['width'])

        corner_size = max(15, min_dim / 5.33333)

        primary_color = (47 / 255, 167 / 255, 215 / 255, 1)
        secondary_color = (148 / 255, 224 / 255, 255 / 255, 1)
        text_color = (255 / 255, 255 / 255, 255 / 255, 1)

        opacity_color = (47 / 255, 167 / 255, 215 / 255, .2)

        x = obj['x']
        y = obj['y']
        w = obj['width']
        h = obj['height']

        # Add Rectangle
        rect = patches.Rectangle((obj['x'], obj['y']), obj['width'], obj['height'], linewidth=1,
                                 edgecolor=opacity_color, facecolor=opacity_color)
        self.axes.add_patch(rect)

        # top left corner
        points = [(x, y + corner_size), (x, y), (x + corner_size, y)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # bottom left corner
        points = [(x, y + h - corner_size), (x, y + h), (x + corner_size, y + h)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # top right corner
        points = [(x + w - corner_size, y), (x + w, y), (x + w, y + corner_size)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # bottom right corner
        points = [(x + w, y + h - corner_size), (x + w, y + h), (x + w - corner_size, y + h)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        padding = max(min_dim * .02, 5)
        corner_size = corner_size - padding

        # 2nd top left corner
        points = [(x + padding, y + padding + corner_size), (x + padding, y + padding),
                  (x + padding + corner_size, y + padding)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # 2nd bottom left corner
        points = [(x + padding, y - padding + h - corner_size), (x + padding, y - padding + h),
                  (x + padding + corner_size, y - padding + h)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # 2nd top right corner
        points = [(x - padding + w - corner_size, y + padding), (x - padding + w, y + padding),
                  (x - padding + w, y + padding + corner_size)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        # 2nd bottom right corner
        points = [(x - padding + w, y - padding + h - corner_size), (x - padding + w, y - padding + h),
                  (x - padding + w - corner_size, y - padding + h)]
        rect = patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False)
        self.axes.add_patch(rect)

        text = plt.text(obj['x'] + 10 + padding, obj['y'] + 10 + padding, label, fontsize=10, color=text_color,
                        horizontalalignment='left', verticalalignment='top')

        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=(1, 1, 1, .7)),
                               path_effects.Stroke(linewidth=1, foreground=(0, 0, 0, .7)), path_effects.Normal()])

    def _label(self, obj: dict) -> str:
        label = obj['classLabel']
        if label == 'person':
            if 'objects' in obj:
                for f in obj['objects']:
                    if 'classLabel' in f and f['classLabel'] == 'face':
                        if 'classes' in f:
                            for c in f['classes']:
                                if 'classLabel' in c:
                                    if c['confidence'] == 1:
                                        label = label + "\n" + c['classLabel']
                                    else:
                                        label = label + "\n" + c['classLabel'] + f" {c['confidence'] * 100:.0f}%" + ""
        return label


# The pose skeletons, mirrored from the Node SDK's eyepop-render-2d/render-pose.ts
# so the 3D plot connects the same joints the 2D renderer does. Keyed on the key
# point group's category and matched by class label, not by index, because the
# label is what identifies a joint across models.
POSE_2D_CATEGORY = "2d-body-points"
POSE_3D_CATEGORY = "3d-body-points"

POSE_2D_CONNECTIONS = (
    ("left shoulder", "right shoulder"),
    ("left hip", "right hip"),
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),
    ("left shoulder", "right hip"),
    ("left hip", "left knee"),
    ("left knee", "left ankle"),
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    ("right shoulder", "left hip"),
    ("right hip", "right knee"),
    ("right knee", "right ankle"),
)

POSE_3D_CONNECTIONS = (
    ("mouth (right)", "mouth (left)"),
    ("right ear", "right eye (outer)"),
    ("right eye (outer)", "right eye"),
    ("right eye", "right eye (inner)"),
    ("right eye (inner)", "nose"),
    ("nose", "left eye (inner)"),
    ("left eye (inner)", "left eye"),
    ("left eye", "left eye (outer)"),
    ("left eye (outer)", "left ear"),
    ("right shoulder", "left shoulder"),
    ("left shoulder", "right hip"),
    ("left hip", "right hip"),
    ("left hip", "right shoulder"),
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    ("right wrist", "right thumb"),
    ("right wrist", "right pinky"),
    ("right wrist", "right index"),
    ("right pinky", "right index"),
    ("left shoulder", "left elbow"),
    ("left elbow", "left wrist"),
    ("left wrist", "left thumb"),
    ("left wrist", "left pinky"),
    ("left wrist", "left index"),
    ("left pinky", "left index"),
    ("right hip", "right knee"),
    ("right knee", "right ankle"),
    ("right ankle", "right foot index"),
    ("right ankle", "right heel"),
    ("right heel", "right foot index"),
    ("left hip", "left knee"),
    ("left knee", "left ankle"),
    ("left ankle", "left foot index"),
    ("left ankle", "left heel"),
    ("left heel", "left foot index"),
)

POSE_CONNECTIONS = {
    POSE_2D_CATEGORY: POSE_2D_CONNECTIONS,
    POSE_3D_CATEGORY: POSE_3D_CONNECTIONS,
}

_NO_POINTS = np.empty((0, 3), dtype="float32")
_NO_SEGMENTS = np.empty((0, 2, 3), dtype="float32")


@dataclass(frozen=True)
class WorldSeries:
    """One set of world coordinates from a prediction, ready to draw.

    `points` are the placed ones, in metres. `segments` are the lines between
    them - a contour's own order, or the pose skeleton for a key point group -
    and are empty for a carrier with no defined connectivity, such as a mask
    point cloud, whose points are a grid rather than a path.
    """

    label: str
    points: np.ndarray
    # a factory because a dataclass refuses an array as a default: it is
    # mutable, even though this one is shared and never written to
    segments: np.ndarray = field(default_factory=lambda: _NO_SEGMENTS)


def _member(obj, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _indexed_points(points) -> np.ndarray:
    """An (N, 3) array of world coordinates, NaN where the worker placed none.

    Index preserving, unlike the placed-only view, because a contour's
    connectivity is its own ordering: dropping an unplaced point first would
    join its neighbours across a gap that is not there.

    A point the worker could not place carries no world members at all - never a
    zero, an Infinity or a NaN - so absence is the only test, and all three
    arrive together.
    """
    rows = []
    for point in points or []:
        x, y, z = _member(point, "worldX"), _member(point, "worldY"), _member(point, "worldZ")
        if x is None or y is None or z is None:
            rows.append((float("nan"),) * 3)
        else:
            rows.append((float(x), float(y), float(z)))
    return np.array(rows, dtype="float32").reshape(-1, 3)


def _placed(indexed: np.ndarray) -> np.ndarray:
    return indexed[~np.isnan(indexed).any(axis=1)]


def _path_segments(indexed: np.ndarray, closed: bool = True) -> np.ndarray:
    """Segments joining consecutive points, skipping any pair with a hole in it.

    An unplaced point breaks the path rather than being bridged over, which is
    what keeps a partially placed contour honest.
    """
    count = len(indexed)
    if count < 2:
        return _NO_SEGMENTS
    starts = range(count if closed else count - 1)
    pairs = [(indexed[i], indexed[(i + 1) % count]) for i in starts]
    kept = [pair for pair in pairs if not (np.isnan(pair[0]).any() or np.isnan(pair[1]).any())]
    return np.array(kept, dtype="float32").reshape(-1, 2, 3)


def _pose_segments(group, indexed: np.ndarray) -> np.ndarray:
    """Skeleton segments for a key point group, or none for an unknown category.

    Matched by class label rather than index, mirroring the 2D renderer: the
    label is what identifies a joint, and an unrecognised category gets no lines
    at all rather than points joined in whatever order they arrived.
    """
    category = _member(group, "category")
    connections = POSE_CONNECTIONS.get(category) if isinstance(category, str) else None
    if not connections:
        return _NO_SEGMENTS

    by_label: dict[str, np.ndarray] = {}
    for point, coordinates in zip(_member(group, "points") or [], indexed, strict=True):
        label = _member(point, "classLabel")
        if label is not None and not np.isnan(coordinates).any():
            by_label[label] = coordinates

    kept = [(by_label[a], by_label[b]) for a, b in connections if a in by_label and b in by_label]
    return np.array(kept, dtype="float32").reshape(-1, 2, 3)


def labelled_world_points(prediction: dict) -> list[WorldSeries]:
    """Every set of world coordinates in a prediction, labelled and connected.

    Covers every carrier the worker enriches - key points, outlines, contours
    (cutouts included), mask point clouds and the scene cloud a `depthMap.toWorld`
    pop returns - rather than only the clouds, so a pop that produces no masks
    still has something to show.

    Labelled by class and carrier, and numbered when a class appears more than
    once, so a plot of several objects can be read. Nested objects are included.
    """
    series: list[WorldSeries] = []
    seen: dict[str, int] = {}

    def add(name: str, carrier: str, indexed: np.ndarray, segments: np.ndarray) -> None:
        placed = _placed(indexed)
        if placed.size:
            series.append(WorldSeries(f"{name} {carrier}", placed, segments))

    def add_group(name: str, group) -> None:
        indexed = _indexed_points(_member(group, "points"))
        add(name, "keypoints", indexed, _pose_segments(group, indexed))

    def walk(objects) -> None:
        for index, obj in enumerate(objects or []):
            name = _member(obj, "classLabel") or f"object {index}"
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} {seen[name]}"

            for group in _member(obj, "keyPoints") or []:
                add_group(name, group)
            outline = _indexed_points(_member(obj, "outline"))
            add(name, "outline", outline, _path_segments(outline))
            for contour in _member(obj, "contours") or []:
                points = _indexed_points(_member(contour, "points"))
                add(name, "contour", points, _path_segments(points))
                for cutout in _member(contour, "cutouts") or []:
                    hole = _indexed_points(cutout)
                    add(name, "cutout", hole, _path_segments(hole))
            cloud = PointCloud.from_object(obj)
            if cloud is not None:
                # a grid rather than a path, so there is nothing to connect
                add(name, "mask", cloud.placed_points, _NO_SEGMENTS)

            walk(_member(obj, "objects"))

    if prediction is None:
        return series

    # a prediction carries key point groups of its own, for the abilities that
    # produce them without an enclosing object
    for group in _member(prediction, "keyPoints") or []:
        indexed = _indexed_points(_member(group, "points"))
        placed = _placed(indexed)
        if placed.size:
            series.append(WorldSeries("keypoints", placed, _pose_segments(group, indexed)))

    walk(_member(prediction, "objects"))

    # last, so the objects a viewer came to look at are not buried under a
    # scene cloud two orders of magnitude larger
    scene = PointCloud.from_depth(_member(prediction, "depth"))
    if scene is not None:
        placed = scene.placed_points
        if placed.size:
            series.append(WorldSeries("scene", placed, _NO_SEGMENTS))
    return series


class EyePopWorldPlot:
    """Scatter of everything in a prediction that carries world coordinates.

    Key points, outlines, contours and mask point clouds alike, in metres.
    Needs a 3D axes - `plt.figure().add_subplot(projection='3d')` - because
    these are real 3D positions rather than an overlay on the frame, which is
    what separates this from EyePopPlot.

    Drawn with Z into the scene and Y up the page rather than in component
    order, since depth reads as distance from the viewer in both frames. See
    AXIS_ORDER; the coordinates themselves are never altered.

    Which frame the metres are in is not recoverable from the prediction. With
    extrinsics they are world coordinates, Z up with the ground at Z = 0;
    without them they are camera coordinates in the OpenCV convention, X right,
    Y **down**, Z forward.

    `invert_y` flips the vertical axis so that a camera frame scene stands the
    right way up: Y grows downwards there, so drawn as-is a figure's feet sit
    above its head. It defaults to on because a source that supplies no
    extrinsics - or identity ones - gets the camera frame, which is the common
    case. Pass `invert_y=False` for coordinates already in a Z-up world frame.
    """

    # A mask is one point per pixel, so a few objects at a few hundred pixels
    # square is already more than a scatter can draw usefully or quickly.
    DEFAULT_MAX_POINTS = 20000

    # Below this a series is drawn whole, however tight the budget. A skeleton
    # of 17 key points and a 40,000 point mask cloud are both series here, and a
    # budget shared evenly between them would thin the skeleton to nothing to
    # save a fraction of the cloud.
    SPARSE_SERIES = 512

    # Which metre component goes on which screen axis: X across, Z into the
    # scene, Y up the page. Matplotlib puts its third axis vertical, and depth
    # is the one component that reads as distance from the viewer in both
    # frames - forward from the camera, or horizontal ground distance under
    # extrinsics - so it belongs in the scene rather than up the page. The data
    # is untouched; only which axis each component is drawn against.
    AXIS_ORDER = (0, 2, 1)
    AXIS_LABELS = ("X (m)", "Z (m)", "Y (m)")

    def __init__(self, axes: Axes3D, invert_y: bool = True):
        self.axes = axes
        self.invert_y = invert_y
        self._plotted = 0
        self._bounds: np.ndarray | None = None

    def prediction(self, prediction: dict, max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot everything in one prediction that has world coordinates.

        Returns the number of points drawn.
        """
        return self.series(labelled_world_points(prediction), max_points)

    def point_clouds(self, clouds: list[PointCloud], labels: list[str] | None = None,
                     max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot mask point clouds only, one colour each."""
        return self.points([cloud.placed_points for cloud in clouds], labels, max_points)

    def points(self, series: list[np.ndarray], labels: list[str] | None = None,
               max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot (N, 3) arrays of world points, one colour each and unconnected."""
        return self.series(
            [WorldSeries(labels[index] if labels is not None and index < len(labels) else "",
                         points)
             for index, points in enumerate(series)],
            max_points)

    def series(self, series: list[WorldSeries], max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot labelled series, drawing each one's connections where it has any.

        The budget is shared across series rather than applied to each, so
        adding objects thins the scatter instead of multiplying it - but sparse
        series are exempt, so key points survive alongside a dense mask.
        """
        drawn = [entry for entry in series if entry.points.size]
        if not drawn:
            return 0

        sparse = sum(len(entry.points) for entry in drawn if self._is_sparse(entry))
        dense = sum(len(entry.points) for entry in drawn if not self._is_sparse(entry))
        budget = max(max_points - sparse, 1) if max_points > 0 else dense
        # strided rather than random so the same call draws the same picture
        stride = max(1, -(-dense // budget)) if dense else 1

        for entry in drawn:
            is_sparse = self._is_sparse(entry)
            sampled = entry.points if is_sparse else entry.points[::stride]
            if not sampled.size:
                continue
            sampled = sampled[:, self.AXIS_ORDER]
            # a handful of key points would be invisible at the size a mask
            # cloud has to be drawn at
            # matplotlib's stub types zs as int; it takes array-like, which is
            # the only shape these can be drawn from
            drawn_points = self.axes.scatter(
                sampled[:, 0], sampled[:, 1],
                zs=sampled[:, 2],  # pyright: ignore[reportArgumentType]
                s=12 if is_sparse else 1,
                alpha=0.9 if is_sparse else 0.5, label=entry.label or None)
            self._plotted += len(sampled)
            self._extend_bounds(sampled)

            # only for a series drawn whole: a strided one has had points removed
            # from under its own connectivity, so its lines would join the wrong
            # pairs. Nothing dense defines any today, so nothing is lost.
            if is_sparse and entry.segments.size:
                self.axes.add_collection3d(Line3DCollection(
                    entry.segments[:, :, self.AXIS_ORDER],
                    colors=drawn_points.get_facecolor(), linewidths=1.5, alpha=0.8))

        return self._plotted

    def _is_sparse(self, entry: "WorldSeries") -> bool:
        return len(entry.points) <= self.SPARSE_SERIES

    def _extend_bounds(self, sampled: np.ndarray) -> None:
        extent = np.stack([sampled.min(axis=0), sampled.max(axis=0)])
        self._bounds = extent if self._bounds is None else np.stack(
            [np.minimum(self._bounds[0], extent[0]), np.maximum(self._bounds[1], extent[1])])

    def finish(self, title: str | None = None, legend: bool = True) -> None:
        """Label the axes and give the box the data's own proportions.

        Metres on every axis, so an unequal box would misrepresent the geometry
        the coordinates exist to measure.
        """
        self.axes.set_xlabel(self.AXIS_LABELS[0])
        self.axes.set_ylabel(self.AXIS_LABELS[1])
        self.axes.set_zlabel(self.AXIS_LABELS[2])
        if title is not None:
            self.axes.set_title(title)
        if self.invert_y and self._bounds is not None:
            # set from the current limits rather than toggled with invert_zaxis,
            # so calling finish twice does not undo it
            low, high = self.axes.get_zlim()
            if low < high:
                self.axes.set_zlim(high, low)
        if self._bounds is not None:
            extents = self._bounds[1] - self._bounds[0]
            # a flat axis would make the box zero thick, which matplotlib rejects
            extents = np.where(extents > 0, extents, 1.0)
            self.axes.set_box_aspect(tuple(extents))
        if legend and self.axes.get_legend_handles_labels()[0]:
            drawn_legend = self.axes.legend(loc="upper right", fontsize="small")
            # one readable size for every entry: a scale factor would blow the
            # sparse series up as far as it lifts the mask dots out of invisibility
            for handle in drawn_legend.legend_handles:
                if handle is not None:
                    handle.set_sizes([24])  # pyright: ignore[reportAttributeAccessIssue]
