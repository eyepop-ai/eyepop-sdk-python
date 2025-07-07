import math
from unicodedata import category

import numpy
import numpy as np
import pyarrow as pa
from eyepop.data.data_types import PredictedObject, UserReview, PredictedClass, PredictedKeyPoint, PredictedKeyPoints, \
    PredictedText, PredictedEmbedding
from pyarrow import Schema

from eyepop.data.arrow.schema import OBJECT_SCHEMA, CLASS_SCHEMA, KEY_POINT_SCHEMA, KEY_POINTS_SCHEMA, \
    TEXT_SCHEMA, EMBEDDING_SCHEMA
from eyepop.data.data_normalize import CONFIDENCE_N_DIGITS, COORDINATE_N_DIGITS


def _round_float_like(float_like: any, digits: int, factor: float = 1.0) -> float | None:
    if float_like is None or np.isnan(float_like):
        return None
    elif type(float_like) is float:
        return round(float_like*factor, digits)
    else:
        return round(float_like.astype(float)*factor, digits)



""" Objects """

def table_from_eyepop_predicted_objects(predicted_objects: list[PredictedObject], source_width: float,
                                        source_height: float, user_review: UserReview | None = None,
                                        schema: Schema = OBJECT_SCHEMA) -> pa.Table:
    classes = []
    confidences = []
    xs = []
    ys = []
    ws = []
    hs = []
    user_reviews = []
    key_pointss = [] if "keyPoints" in schema.names else None
    categories = [] if "category" in schema.names else None
    texts = [] if "texts" in schema.names else None
    for o in predicted_objects:
        classes.append(o.classLabel)
        confidences.append(round(o.confidence, CONFIDENCE_N_DIGITS) if o.confidence is not None else None)
        xs.append(numpy.float16(round(o.x / source_width, COORDINATE_N_DIGITS)))
        ys.append(numpy.float16(round(o.y / source_height, COORDINATE_N_DIGITS)))
        ws.append(numpy.float16(round(o.width / source_width, COORDINATE_N_DIGITS)))
        hs.append(numpy.float16(round(o.height / source_height, COORDINATE_N_DIGITS)))
        user_reviews.append(user_review)
        if key_pointss is not None:
            if o.keyPoints is not None:
                key_pointss.append(table_from_eyepop_predicted_key_pointss(
                    o.keyPoints, source_width, source_height,
                    schema=pa.schema(schema.field(7).type.value_type),  # schema for "keyPoints" field
                ).to_pylist())
            else:
                key_pointss.append(None)
        if categories is not None:
            categories.append(o.category)
        if texts is not None:
            if o.texts is not None:
                texts.append(table_from_eyepop_predicted_texts(
                    o.texts,
                    schema=pa.schema(schema.field(9).type.value_type),  # schema for "texts" field
                ).to_pylist())
            else:
                texts.append(None)

    columns = [
        pa.array(classes).dictionary_encode(),
        pa.array(confidences),
        pa.array(xs), pa.array(ys), pa.array(ws), pa.array(hs),
        pa.array(user_reviews),
    ]
    # since v1.1
    if key_pointss is not None:
        columns.append(key_pointss)
    # since v1.2
    if categories is not None:
        columns.append(pa.array(categories).dictionary_encode())
    if texts is not None:
        columns.append(texts)
    return pa.Table.from_arrays(columns, schema=schema)


def eyepop_predicted_objects_from_table(
        table: pa.Table, source_width: float, source_height: float
) -> list[PredictedObject]:
    if table.schema != OBJECT_SCHEMA:
        raise ValueError(f"expected table schema {OBJECT_SCHEMA}")
    predicted_objects = eyepop_predicted_objects_from_pylist(table.to_pylist(), source_height, source_width)

    return predicted_objects


def eyepop_predicted_objects_from_pylist(py_list: list[dict[str, any]],
                                         source_height: float,
                                         source_width: float) -> list[PredictedObject]:
    predicted_objects: list[PredictedObject | None] = [None] * len(py_list)
    for i, o in enumerate(py_list):
        confidence = _round_float_like(o.get("confidence", None), CONFIDENCE_N_DIGITS)
        key_pointss_as_pylist = o.get("keyPoints", None)
        if key_pointss_as_pylist is not None:
            key_pointss = eyepop_predicted_key_pointss_from_pylist(key_pointss_as_pylist, source_height, source_width)
        else:
            key_pointss = None
        texts_as_pylist = o.get("texts", None)
        if texts_as_pylist is None:
            child_texts = None
        else:
            child_texts = eyepop_predicted_texts_from_pylist(texts_as_pylist)
        predicted_objects[i] = PredictedObject(
            classLabel=o["classLabel"],
            confidence=confidence,
            x=_round_float_like(o["x"], COORDINATE_N_DIGITS, 1/source_width),
            y=_round_float_like(o["y"], COORDINATE_N_DIGITS, 1/source_height),
            width=_round_float_like(o["width"], COORDINATE_N_DIGITS, 1/source_width),
            height=_round_float_like(o["height"], COORDINATE_N_DIGITS, 1/source_height),
            keyPoints=key_pointss,
            category=o.get("category", None),
            texts=child_texts,
        )
    return predicted_objects


""" Classes """

def table_from_eyepop_predicted_classes(predicted_classes: list[PredictedClass],
                                        user_review: UserReview | None = None,
                                        schema: Schema = CLASS_SCHEMA) -> pa.Table:
    classes = []
    confidences = []
    user_reviews = []
    categories = []

    for o in predicted_classes:
        classes.append(o.classLabel)
        confidences.append(round(o.confidence, CONFIDENCE_N_DIGITS))
        user_reviews.append(user_review)
        categories.append(o.category)

    columns = [
        pa.array(classes).dictionary_encode(),
        pa.array(confidences),
        pa.array(user_reviews),
    ]
    # since v1.2
    if "category" in schema.names:
        columns.append(pa.array(categories).dictionary_encode())

    return pa.Table.from_arrays(columns, schema=schema)


def eyepop_predicted_classes_from_table(table: pa.Table) -> list[PredictedClass] | None:
    if table.schema != CLASS_SCHEMA:
        raise ValueError(f"expected table schema {CLASS_SCHEMA}")
    predicted_classes = eyepop_predicted_classes_from_pylist(table.to_pylist())

    return predicted_classes


def eyepop_predicted_classes_from_pylist(py_list: list[dict[str, any]]) -> list[PredictedClass] | None:
    if py_list is None:
        return None
    predicted_classes = []
    for i, o in enumerate(py_list):
        predicted_classes.append(PredictedClass(
            classLabel=o["classLabel"],
            confidence=_round_float_like(o.get("confidence", None), CONFIDENCE_N_DIGITS),
            category=o.get("category", None)
        ))
    return predicted_classes

""" Texts """

def table_from_eyepop_predicted_texts(predicted_texts: list[PredictedText], schema: Schema = TEXT_SCHEMA) -> pa.Table:
    confidences = []
    texts = []
    categories = []
    for predicted_text in predicted_texts:
        confidences.append(round(predicted_text.confidence, CONFIDENCE_N_DIGITS) if predicted_text.confidence is not None else None)
        texts.append(predicted_text.text)
        categories.append(predicted_text.category)
    return pa.Table.from_arrays([
        pa.array(confidences),
        texts,
        pa.array(categories).dictionary_encode(),
    ], schema=schema)

def eyepop_predicted_texts_from_pylist(py_list: list[dict[str, any]]) -> list[PredictedText]:
    predicted_texts: list[PredictedText | None] = [None] * len(py_list)
    for i, predicted_text in enumerate(py_list):
        predicted_texts[i] = PredictedText(
            text=predicted_text["text"],
            confidence=_round_float_like(predicted_text.get("confidence", None), CONFIDENCE_N_DIGITS),
            category=predicted_text.get("category", None),
        )
    return predicted_texts


""" Key Pointss [plural] """

def table_from_eyepop_predicted_key_pointss(predicted_key_pointss: list[PredictedKeyPoints],
                                            source_width: float,
                                            source_height: float,
                                            schema: Schema = KEY_POINTS_SCHEMA) -> pa.Table:
    types = []
    key_points = []
    categories = []
    for kps in predicted_key_pointss:
        types.append(kps.type)
        if kps.points is not None:
            key_points.append(table_from_eyepop_predicted_key_points(
                kps.points, source_width, source_height,
                schema=pa.schema(schema.field(1).type.value_type),  # schema for "points" field
            ).to_pylist())
        else:
            key_points.append(None)
        categories.append(kps.category)

    columns = [
        pa.array(types).dictionary_encode(),
        key_points
    ]
    if "category" in schema.names:
        columns.append(pa.array(categories).dictionary_encode())
    return pa.Table.from_arrays(columns, schema=schema)


def eyepop_predicted_key_pointss_from_table(
        table: pa.Table, source_width: float, source_height: float
) -> list[PredictedKeyPoints]:
    if table.schema != KEY_POINTS_SCHEMA:
        raise ValueError(f"expected table schema {KEY_POINTS_SCHEMA}")
    predicted_key_pointss = eyepop_predicted_key_pointss_from_pylist(table.to_pylist(), source_height, source_width)

    return predicted_key_pointss


def eyepop_predicted_key_pointss_from_pylist(py_list: list[dict[str, any]],
                                             source_height: float,
                                             source_width: float) -> list[PredictedKeyPoints]:
    predicted_key_pointss: list[PredictedKeyPoints | None] = [None] * len(py_list)
    for i, kps in enumerate(py_list):
        predicted_key_pointss[i] = PredictedKeyPoints(
            type=kps.get("type", None),
            points=eyepop_predicted_key_points_from_pylist(kps["points"], source_height, source_width) if kps["points"] is not None else None,
            category=kps.get("category", None),
        )
    return predicted_key_pointss

""" Key Points """

def table_from_eyepop_predicted_key_points(predicted_key_points: list[PredictedKeyPoint],
                                           source_width: float,
                                           source_height: float,
                                           schema: Schema = KEY_POINT_SCHEMA) -> pa.Table:
    classes = []
    confidences = []
    xs = []
    ys = []
    zs = []
    visibles = []
    categories = []
    for kp in predicted_key_points:
        classes.append(kp.classLabel)
        confidences.append(round(kp.confidence, CONFIDENCE_N_DIGITS) if kp.confidence is not None else None)
        xs.append(numpy.float16(round(kp.x / source_width, COORDINATE_N_DIGITS)))
        ys.append(numpy.float16(round(kp.y / source_height, COORDINATE_N_DIGITS)))
        zs.append(numpy.float16(round(kp.z / max(source_width, source_height), COORDINATE_N_DIGITS)) if kp.z is not None else None)
        visibles.append(kp.visible)
        categories.append(kp.category)

    columns = [
        pa.array(classes).dictionary_encode(),
        pa.array(confidences),
        pa.array(xs), pa.array(ys), pa.array(zs),
        pa.array(visibles)
    ]
    if "category" in schema.names:
        columns.append(pa.array(categories).dictionary_encode())

    return pa.Table.from_arrays(columns, schema=schema)


def eyepop_predicted_key_points_from_table(
        table: pa.Table, source_width: float, source_height: float
) -> list[PredictedKeyPoint]:
    if table.schema != KEY_POINT_SCHEMA:
        raise ValueError(f"expected table schema {KEY_POINT_SCHEMA}")
    predicted_key_points = eyepop_predicted_key_points_from_pylist(table.to_pylist(), source_height, source_width)
    return predicted_key_points


def eyepop_predicted_key_points_from_pylist(py_list: list[dict[str, any]],
                                            source_height: float,
                                            source_width: float) -> list[PredictedKeyPoint]:
    predicted_key_points: list[PredictedKeyPoint | None] = [None] * len(py_list)
    for i, kp in enumerate(py_list):
        confidence = _round_float_like(kp.get("confidence", None), CONFIDENCE_N_DIGITS)
        predicted_key_points[i] = PredictedKeyPoint(
            classLabel=kp.get("classLabel", None),
            confidence=confidence,
            x=_round_float_like(kp["x"], COORDINATE_N_DIGITS, 1/source_width),
            y=_round_float_like(kp["y"], COORDINATE_N_DIGITS, 1/source_height),
            z=_round_float_like(kp["z"], COORDINATE_N_DIGITS, 1/max(source_width, source_height)) if kp.get("z", None) is not None else None,
            visible=kp.get("visible", None),
            category=kp.get("category", None),
        )
    return predicted_key_points

""" Embeddings """
def table_from_eyepop_predicted_embeddings(predicted_embeddings: list[PredictedEmbedding],
                                           schema: Schema = EMBEDDING_SCHEMA) -> pa.Table:
    embeddings = []
    x_coordinates = []
    y_coordinates = []
    for predicted_embedding in predicted_embeddings:
        embeddings.append(predicted_embedding.embedding)
        x_coordinates.append(predicted_embedding.x)
        y_coordinates.append(predicted_embedding.y)
    return pa.Table.from_arrays([
        pa.array(embeddings),
        pa.array(x_coordinates),
        pa.array(y_coordinates),
    ], schema=schema)

def eyepop_predicted_embeddings_from_pylist(py_list: list[dict[str, any]]) -> list[PredictedEmbedding]:
    predicted_embeddings: list[PredictedEmbedding | None] = [None] * len(py_list)
    for i, predicted_embedding in enumerate(py_list):
        predicted_embeddings[i] = PredictedEmbedding(
            embedding=predicted_embedding["embedding"],
            x=_round_float_like(predicted_embedding.get("x", None), COORDINATE_N_DIGITS),
            y=_round_float_like(predicted_embedding.get("y", None), COORDINATE_N_DIGITS),
        )
    return predicted_embeddings