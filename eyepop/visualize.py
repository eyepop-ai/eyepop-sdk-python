import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class EyePopPlot:
    def __init__(self, axes: Axes):
        self.axes = axes

    def prediction(self, prediction: dict | None) -> None:
        if prediction is None:
            return
        objects = prediction.get('objects', None)
        if objects is not None:
            for obj in objects:
                self.object(obj)

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
        background = patches.Rectangle((obj['x'], obj['y']), obj['width'], obj['height'], linewidth=1,
                                 edgecolor=opacity_color, facecolor=opacity_color)
        self.axes.add_patch(background)

        # top left corner
        points = [(x, y + corner_size), (x, y), (x + corner_size, y)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False))

        # bottom left corner
        points = [(x, y + h - corner_size), (x, y + h), (x + corner_size, y + h)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False))

        # top right corner
        points = [(x + w - corner_size, y), (x + w, y), (x + w, y + corner_size)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False))

        # bottom right corner
        points = [(x + w, y + h - corner_size), (x + w, y + h), (x + w - corner_size, y + h)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=primary_color, facecolor='none', closed=False))

        padding = max(min_dim * .02, 5)
        corner_size = corner_size - padding

        # 2nd top left corner
        points = [(x + padding, y + padding + corner_size), (x + padding, y + padding),
                  (x + padding + corner_size, y + padding)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False))

        # 2nd bottom left corner
        points = [(x + padding, y - padding + h - corner_size), (x + padding, y - padding + h),
                  (x + padding + corner_size, y - padding + h)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False))

        # 2nd top right corner
        points = [(x - padding + w - corner_size, y + padding), (x - padding + w, y + padding),
                  (x - padding + w, y + padding + corner_size)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False))

        # 2nd bottom right corner
        points = [(x - padding + w, y - padding + h - corner_size), (x - padding + w, y - padding + h),
                  (x - padding + w - corner_size, y - padding + h)]
        self.axes.add_patch(patches.Polygon(points, linewidth=1, edgecolor=secondary_color, facecolor='none', closed=False))

        text = plt.text(obj['x'] + 10 + padding, obj['y'] + 10 + padding, label, fontsize=10, color=text_color,
                        horizontalalignment='left', verticalalignment='top')

        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=(1, 1, 1, .7)),
                               path_effects.Stroke(linewidth=1, foreground=(0, 0, 0, .7)), path_effects.Normal()])

    def _label(self, obj: dict) -> str:
        label: str = obj['classLabel']
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
