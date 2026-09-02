import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

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


def _world_points(points) -> np.ndarray:
    """The placed points of a point list, as an (N, 3) array in metres.

    A point the worker could not place carries no world members at all - never a
    zero, an Infinity or a NaN - so absence is the only test, and all three
    arrive together.
    """
    placed = []
    for point in points or []:
        if not isinstance(point, dict):
            point = {"worldX": getattr(point, "worldX", None),
                     "worldY": getattr(point, "worldY", None),
                     "worldZ": getattr(point, "worldZ", None)}
        x, y, z = point.get("worldX"), point.get("worldY"), point.get("worldZ")
        if x is not None and y is not None and z is not None:
            placed.append((float(x), float(y), float(z)))
    return np.array(placed, dtype="float32").reshape(-1, 3)


def labelled_world_points(prediction: dict) -> list[tuple[str, np.ndarray]]:
    """Every set of world coordinates in a prediction, labelled for a legend.

    Covers all four carriers the worker enriches - key points, outlines,
    contours (cutouts included) and mask point clouds - rather than only the
    clouds, so a pop that produces no masks still has something to show.

    Each entry is an (N, 3) array of metres. Labelled by class and carrier, and
    numbered when a class appears more than once, so a plot of several objects
    can be read. Nested objects are included.
    """
    labelled: list[tuple[str, np.ndarray]] = []
    seen: dict[str, int] = {}

    def add(name: str, carrier: str, points: np.ndarray) -> None:
        if points.size:
            labelled.append((f"{name} {carrier}", points))

    def member(obj, key: str):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def walk(objects, depth: int) -> None:
        for index, obj in enumerate(objects or []):
            name = member(obj, "classLabel") or f"object {index}"
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} {seen[name]}"

            for group in member(obj, "keyPoints") or []:
                add(name, "keypoints", _world_points(member(group, "points")))
            add(name, "outline", _world_points(member(obj, "outline")))
            for contour in member(obj, "contours") or []:
                add(name, "contour", _world_points(member(contour, "points")))
                for cutout in member(contour, "cutouts") or []:
                    add(name, "cutout", _world_points(cutout))
            cloud = PointCloud.from_object(obj)
            if cloud is not None:
                add(name, "mask", cloud.placed_points)

            walk(member(obj, "objects"), depth + 1)

    if prediction is None:
        return labelled

    # a prediction carries key point groups of its own, for the abilities that
    # produce them without an enclosing object
    for group in (prediction.get("keyPoints") if isinstance(prediction, dict)
                  else getattr(prediction, "keyPoints", None)) or []:
        points = _world_points(group.get("points") if isinstance(group, dict)
                               else getattr(group, "points", None))
        if points.size:
            labelled.append(("keypoints", points))

    walk(prediction.get("objects") if isinstance(prediction, dict)
         else getattr(prediction, "objects", None), 0)
    return labelled


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
        labelled = labelled_world_points(prediction)
        return self.points([points for _, points in labelled],
                           labels=[label for label, _ in labelled],
                           max_points=max_points)

    def point_clouds(self, clouds: list[PointCloud], labels: list[str] | None = None,
                     max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot mask point clouds only, one colour each."""
        return self.points([cloud.placed_points for cloud in clouds], labels, max_points)

    def points(self, series: list[np.ndarray], labels: list[str] | None = None,
               max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot (N, 3) arrays of world points, one colour each.

        The budget is shared across series rather than applied to each, so
        adding objects thins the scatter instead of multiplying it - but sparse
        series are exempt, so key points survive alongside a dense mask.
        """
        drawn = [(index, points) for index, points in enumerate(series) if points.size]
        if not drawn:
            return 0

        sparse = sum(len(points) for _, points in drawn if len(points) <= self.SPARSE_SERIES)
        dense = sum(len(points) for _, points in drawn if len(points) > self.SPARSE_SERIES)
        budget = max(max_points - sparse, 1) if max_points > 0 else dense
        # strided rather than random so the same call draws the same picture
        stride = max(1, -(-dense // budget)) if dense else 1

        for index, points in drawn:
            is_sparse = len(points) <= self.SPARSE_SERIES
            sampled = points if is_sparse else points[::stride]
            if not sampled.size:
                continue
            sampled = sampled[:, self.AXIS_ORDER]
            label = labels[index] if labels is not None and index < len(labels) else None
            # a handful of key points would be invisible at the size a mask
            # cloud has to be drawn at
            # matplotlib's stub types zs as int; it takes array-like, which is
            # the only shape these can be drawn from
            self.axes.scatter(sampled[:, 0], sampled[:, 1],
                              zs=sampled[:, 2],  # pyright: ignore[reportArgumentType]
                              s=12 if is_sparse else 1,
                              alpha=0.9 if is_sparse else 0.5, label=label)
            self._plotted += len(sampled)
            extent = np.stack([sampled.min(axis=0), sampled.max(axis=0)])
            self._bounds = extent if self._bounds is None else np.stack(
                [np.minimum(self._bounds[0], extent[0]), np.maximum(self._bounds[1], extent[1])])

        return self._plotted

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
