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


def labelled_point_clouds(prediction: dict) -> list[tuple[str, PointCloud]]:
    """Every point cloud in a prediction, paired with a label for a legend.

    Labelled by class, numbered when a class appears more than once, so a plot
    of several objects can be read. Nested objects are included, in the same
    outermost-first order PointCloud.from_prediction uses.
    """
    labelled: list[tuple[str, PointCloud]] = []
    seen: dict[str, int] = {}

    def walk(objects) -> None:
        for index, obj in enumerate(objects or []):
            cloud = PointCloud.from_object(obj)
            if cloud is not None:
                name = (obj.get("classLabel") if isinstance(obj, dict)
                        else getattr(obj, "classLabel", None)) or f"object {index}"
                seen[name] = seen.get(name, 0) + 1
                labelled.append((f"{name} {seen[name]}" if seen[name] > 1 else name, cloud))
            walk(obj.get("objects") if isinstance(obj, dict) else getattr(obj, "objects", None))

    if prediction is None:
        return labelled
    walk(prediction.get("objects") if isinstance(prediction, dict)
         else getattr(prediction, "objects", None))
    return labelled


class EyePopPointCloudPlot:
    """Scatter of per-object mask point clouds, in metres.

    Needs a 3D axes - `plt.figure().add_subplot(projection='3d')` - because the
    points are a real 3D cloud rather than an overlay on the frame, which is
    what separates this from EyePopPlot.

    Which frame the metres are in depends on the source's calibration and is not
    recoverable from the prediction: with extrinsics they are world coordinates,
    Z up with the ground at Z = 0; without them they are camera coordinates in
    the OpenCV convention, X right, Y **down**, Z forward. The axes are labelled
    but not reoriented, since guessing wrong would silently flip the scene.
    """

    # A mask is one point per pixel, so a few objects at a few hundred pixels
    # square is already more than a scatter can draw usefully or quickly.
    DEFAULT_MAX_POINTS = 20000

    def __init__(self, axes: Axes3D):
        self.axes = axes
        self._plotted = 0
        self._bounds: np.ndarray | None = None

    def prediction(self, prediction: dict, max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot every point cloud in one prediction. Returns the points drawn."""
        labelled = labelled_point_clouds(prediction)
        return self.point_clouds([cloud for _, cloud in labelled],
                                 labels=[label for label, _ in labelled],
                                 max_points=max_points)

    def point_clouds(self, clouds: list[PointCloud], labels: list[str] | None = None,
                     max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot several clouds, one colour each, sharing a single point budget.

        The budget is shared rather than per cloud so that adding objects thins
        the scatter instead of multiplying it.
        """
        placed = [cloud.placed_points for cloud in clouds]
        return self.points([points for points in placed if points.size], labels, max_points)

    def points(self, clouds: list[np.ndarray], labels: list[str] | None = None,
               max_points: int = DEFAULT_MAX_POINTS) -> int:
        """Plot (N, 3) arrays of already placed points, one colour each."""
        clouds = [points for points in clouds if points.size]
        if not clouds:
            return 0

        total = sum(len(points) for points in clouds)
        # strided rather than random so the same call draws the same picture
        stride = max(1, -(-total // max_points)) if max_points > 0 else 1

        for index, points in enumerate(clouds):
            sampled = points[::stride]
            if not sampled.size:
                continue
            label = labels[index] if labels is not None and index < len(labels) else None
            # matplotlib's stub types zs as int; it takes array-like, which is
            # the only shape a cloud can be drawn from
            self.axes.scatter(sampled[:, 0], sampled[:, 1],
                              zs=sampled[:, 2],  # pyright: ignore[reportArgumentType]
                              s=1, alpha=0.5, label=label)
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
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")
        self.axes.set_zlabel("Z (m)")
        if title is not None:
            self.axes.set_title(title)
        if self._bounds is not None:
            extents = self._bounds[1] - self._bounds[0]
            # a flat axis would make the box zero thick, which matplotlib rejects
            extents = np.where(extents > 0, extents, 1.0)
            self.axes.set_box_aspect(tuple(extents))
        if legend and self.axes.get_legend_handles_labels()[0]:
            self.axes.legend(loc="upper right", fontsize="small", markerscale=8)
