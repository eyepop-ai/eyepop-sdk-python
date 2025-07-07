from eyepop.data.data_types import AssetAnnotationResponse, Prediction, PredictedObject, PredictedClass

# Confidence and coordinates are represented as float16 in the arrow format but Python lacks support for 2-byte floats.
# To avoid "changing" 4-bytes floats when converted back and forth, we will always round to these precisions.

CONFIDENCE_N_DIGITS = 3
COORDINATE_N_DIGITS = 3

def normalize_eyepop_annotations(
        annotations: list[AssetAnnotationResponse]
):
    for annotation in annotations:
        if annotation.annotation is not None:
            normalize_eyepop_prediction(annotation.annotation)


def normalize_eyepop_prediction(
        prediction: Prediction
):
    if prediction.objects:
        normalize_predicted_objects(
            prediction.objects, prediction.source_width, prediction.source_height)
    if prediction.classes:
        normalize_predicted_classes(prediction.classes)
    prediction.source_width = 1.0
    prediction.source_height = 1.0


def normalize_predicted_objects(
        predicted_objects: list[PredictedObject],
        original_source_width: float,
        original_source_height: float
):
    for o in predicted_objects:
        if o.objects:
            normalize_predicted_objects(o.objects, original_source_width, original_source_height)
        if o.classes:
            normalize_predicted_classes(o.classes)
        o.x = round(o.x / original_source_width, COORDINATE_N_DIGITS)
        o.y = round(o.y / original_source_height, COORDINATE_N_DIGITS)
        o.width = round(o.width / original_source_width, COORDINATE_N_DIGITS)
        o.height = round(o.height / original_source_height, COORDINATE_N_DIGITS)
        if o.confidence is not None:
            o.confidence = round(o.confidence, CONFIDENCE_N_DIGITS)


def normalize_predicted_classes(
        predicted_classes: list[PredictedClass]
):
    for c in predicted_classes:
        if c.confidence is not None:
            c.confidence = round(c.confidence, CONFIDENCE_N_DIGITS)
