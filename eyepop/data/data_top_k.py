import heapq
from typing import TypeVar

from eyepop.data.data_types import PredictedClass, PredictedObject, Prediction

T = TypeVar("T", bound=PredictedClass)


def _top_k(items: list[T], k: int) -> list[T]:
    """Find k items with the top confidence."""

    if len(items) <= k:
        return items

    # extract (comparison value, index, value) as heapifyable items, using i as a tiebreaker if confidences are equal
    heap_items: list[tuple[float, int, T]] = [
        ((item.confidence if item.confidence is not None else -1.0), i, item)
        for i, item in enumerate(items)
    ]

    # Create a min-heap with the first k elements
    min_heap = heap_items[:k]
    heapq.heapify(min_heap)

    # Traverse the rest of the array
    for x in heap_items[k:]:
        if x[0] > min_heap[0][0]:
            heapq.heappushpop(min_heap, x)

    res: list[tuple[float, int, T]] = []

    # Min heap will contain only k
    # largest element
    while min_heap:
        res.append(heapq.heappop(min_heap))

    # Reverse the result array, so that all
    # elements are in decreasing order
    res.reverse()

    return [item[2] for item in res]


def filter_prediction_top_k(prediction: Prediction, k: int) -> Prediction:
    objects = prediction.objects
    classes = prediction.classes
    if (objects is None or len(objects) <= k) and (classes is None or len(classes) <= k):
        return prediction

    top_objects: list[PredictedObject] | None = _top_k(objects, k) if objects is not None else None
    top_classes: list[PredictedClass] | None = _top_k(classes, k) if classes is not None else None

    return Prediction(
        source_width=prediction.source_width,
        source_height=prediction.source_height,
        objects=top_objects,
        classes=top_classes,
        texts=prediction.texts,
        meshs=prediction.meshs,
        keyPoints=prediction.keyPoints,
    )
