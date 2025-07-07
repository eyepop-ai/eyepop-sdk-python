
import pyarrow as pa
from eyepop.data.data_types import AssetAnnotationResponse, Prediction

from eyepop.data.arrow.eyepop.predictions import table_from_eyepop_predicted_objects, \
    eyepop_predicted_objects_from_pylist, \
    eyepop_predicted_classes_from_pylist, table_from_eyepop_predicted_classes, table_from_eyepop_predicted_key_pointss, \
    eyepop_predicted_key_pointss_from_pylist, table_from_eyepop_predicted_texts, eyepop_predicted_texts_from_pylist, \
    eyepop_predicted_embeddings_from_pylist, table_from_eyepop_predicted_embeddings
from eyepop.data.arrow.schema import ANNOTATION_SCHEMA
from eyepop.data.arrow.schema_version_conversion import convert


def table_from_eyepop_annotations(annotations: list[AssetAnnotationResponse], schema: pa.Schema = ANNOTATION_SCHEMA) -> pa.Table:
    types = []
    sources = []
    user_reviews = []
    objects = []
    classes = []
    source_model_uuids = [] if "source_model_uuid" in schema.names else None
    key_points = [] if "keyPoints" in schema.names else None
    texts = [] if "texts" in schema.names else None
    embeddings = [] if "embeddings" in schema.names else None

    for e in annotations:
        types.append(e.type)
        sources.append(e.source)
        user_reviews.append(e.user_review)
        if e.annotation.objects is None:
            objects.append(None)
        elif len(e.annotation.objects) == 0:
            objects.append([])
        else:
            objects.append(table_from_eyepop_predicted_objects(
                e.annotation.objects,
                e.annotation.source_width,
                e.annotation.source_height,
                e.user_review,
                schema=pa.schema(schema.field(3).type.value_type), # schema for "objects" field
            ).to_struct_array())
        if e.annotation.classes is None:
            classes.append(None)
        elif len(e.annotation.classes) == 0:
            classes.append([])
        else:
            classes.append(table_from_eyepop_predicted_classes(
                e.annotation.classes,
                e.user_review,
                schema=pa.schema(schema.field(4).type.value_type),  # schema for "classes" field
            ).to_struct_array())
        if source_model_uuids is not None:
            source_model_uuids.append(e.source_model_uuid)
        if key_points is not None:
            if e.annotation.keyPoints is None:
                key_points.append(None)
            elif len(e.annotation.keyPoints) == 0:
                key_points.append([])
            else:
                key_points.append(table_from_eyepop_predicted_key_pointss(
                    e.annotation.keyPoints,
                    e.annotation.source_width,
                    e.annotation.source_height,
                    schema=pa.schema(schema.field(6).type.value_type),  # schema for "keyPoints" field
                ).to_struct_array())
        if texts is not None:
            if e.annotation.texts is None:
                texts.append(None)
            elif len(e.annotation.texts) == 0:
                texts.append([])
            else:
                texts.append(table_from_eyepop_predicted_texts(e.annotation.texts).to_struct_array())
        if embeddings is not None:
            if e.annotation.embeddings is None:
                embeddings.append(None)
            elif len(e.annotation.embeddings) == 0:
                embeddings.append([])
            else:
                embeddings.append(table_from_eyepop_predicted_embeddings(
                    e.annotation.embeddings
                ).to_struct_array())

    # since 0.0
    columns = [
        pa.array(types).dictionary_encode(),
        pa.array(sources).dictionary_encode(),
        pa.array(user_reviews).dictionary_encode(),
        objects,
        classes,
    ]

    # since 1.0: source_model_uuid
    if source_model_uuids is not None:
        columns.append(pa.array(source_model_uuids).dictionary_encode())

    # since 1.1: key_points
    if key_points is not None:
        columns.append(key_points)

    # since 1.2: texts
    if texts is not None:
        columns.append(texts)

    # since 1.3: embeddings
    if embeddings is not None:
        columns.append(embeddings)

    return pa.Table.from_arrays(columns, schema=schema)


def eyepop_annotations_from_table(table: pa.Table) -> list[AssetAnnotationResponse]:
    table = convert(table, ANNOTATION_SCHEMA)
    annotations = []
    i = 0
    for batch in table.to_reader():
        types = batch.column(0).to_pylist()
        sources = batch.column(1).to_pylist()
        user_reviews = batch.column(2).to_pylist()
        objects = batch.column(3).to_pylist()
        classes = batch.column(4).to_pylist()
        # since 1.0: source_model_uuid
        source_model_uuid = batch.column(5).to_pylist()
        # since 1.1: key_points
        key_points = batch.column(6).to_pylist()

        for j in range(len(types)):
            if objects[j] is None:
                child_objects = None
            elif len(objects[j]) == 0:
                child_objects = []
            else:
                child_objects = eyepop_predicted_objects_from_pylist(objects[j], 1.0, 1.0)

            if classes[j] is None:
                child_classes = None
            elif len(classes[j]) == 0:
                child_classes = []
            else:
                child_classes = eyepop_predicted_classes_from_pylist(classes[j])

            if key_points[j] is None:
                child_key_pointss = None
            elif len(key_points[j]) == 0:
                child_key_pointss = []
            else:
                child_key_pointss = eyepop_predicted_key_pointss_from_pylist(key_points[j], 1.0, 1.0)

            annotations.append(AssetAnnotationResponse(
                type=types[j],
                user_review=user_reviews[j],
                source=sources[j],
                annotation=Prediction(
                    source_width=1.0,
                    source_height=1.0,
                    objects=child_objects,
                    classes=child_classes,
                    keyPoints=child_key_pointss
                ),
                source_model_uuid=source_model_uuid[j],
            ))
            i += 1
    return annotations


def eyepop_annotations_from_pylist(py_list: list[dict]) -> list[AssetAnnotationResponse]:
    annotations = []
    for i, o in enumerate(py_list):
        if o is None:
            continue
        objects = o.get('objects', None)
        if objects is None:
            child_objects = None
        else:
            child_objects = eyepop_predicted_objects_from_pylist(objects, 1.0, 1.0)
        classes = o.get('classes', None)
        if classes is None:
            child_classes = None
        else:
            child_classes = eyepop_predicted_classes_from_pylist(classes)
        key_pointss = o.get("keyPoints", None)
        if key_pointss is None:
            child_key_pointss = None
        else:
            child_key_pointss = eyepop_predicted_key_pointss_from_pylist(key_pointss, 1.0, 1.0)
        texts = o.get("texts", None)
        if texts is None:
            child_texts = None
        else:
            child_texts = eyepop_predicted_texts_from_pylist(texts)
        embeddings = o.get("embeddings", None)
        if embeddings is None:
            child_embeddings = None
        else:
            child_embeddings = eyepop_predicted_embeddings_from_pylist(embeddings)
        annotations.append(AssetAnnotationResponse(
            type=o['type'],
            user_review=o['user_review'],
            source=o['source'],
            annotation=Prediction(
                source_width=1.0,
                source_height=1.0,
                objects=child_objects,
                classes=child_classes,
                keyPoints=child_key_pointss,
                texts=child_texts,
                embeddings=child_embeddings
            ),
            source_model_uuid = o.get('source_model_uuid', None)
        ))
    return annotations
