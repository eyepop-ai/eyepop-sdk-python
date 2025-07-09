from eyepop.data.data_types import Prediction, PredictedClass

import heapq

def _top_k(classes: list[PredictedClass], k: int) -> list[PredictedClass]:
    """ Find k classes with the top confidence """

    if len(classes) <= k:
        return classes

    # extract (comparison value, value) as heapifyalbe items, using i asa tiebreaker if confidences are equal
    classes = [((clazz.confidence if clazz.confidence is not None else -1.0), i, clazz) for i, clazz in enumerate(classes)]

    # Create a min-heap with the first k elements
    min_heap = classes[:k]
    heapq.heapify(min_heap)

    # Traverse the rest of the array
    for x in classes[k:]:
        if x[0] > min_heap[0][0]:
            heapq.heappushpop(min_heap, x)

    res = []

    # Min heap will contain only k
    # largest element
    while min_heap:
        res.append(heapq.heappop(min_heap))

    # Reverse the result array, so that all
    # elements are in decreasing order
    res.reverse()

    return [i[2] for i in res]

def filter_prediction_top_k(prediction: Prediction, k: int) -> Prediction:
    objects = prediction.objects
    classes = prediction.classes
    if (objects is None or len(objects) <= k) and (classes is None or len(classes) <= k):
        return prediction
    return Prediction(
        source_width=prediction.source_width,
        source_height=prediction.source_height,
        objects=_top_k(objects, k) if objects is not None else None,
        classes=_top_k(classes, k) if classes is not None else None,
        texts=prediction.texts,
        meshs=prediction.meshs,
        keyPoints=prediction.keyPoints,
    )
