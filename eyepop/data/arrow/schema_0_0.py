import pyarrow as pa

from eyepop.data.data_types import MIME_TYPE_APACHE_ARROW_FILE

""" Arrow schema for Asset export/import form Data API. """

MIME_TYPE_APACHE_ARROW_FILE_VERSIONED = f"{MIME_TYPE_APACHE_ARROW_FILE};version=0.0"

_object_fields = [
    # Python type is string, there should be only as small number of
    # classLabel values per prediction, hence the dictionary encoding.
    pa.field("classLabel", pa.dictionary(pa.int32(), pa.string())),
    pa.field("confidence", pa.float16()),
    pa.field("x", pa.float16()),
    pa.field("y", pa.float16()),
    pa.field("width", pa.float16()),
    pa.field("height", pa.float16()),
    # from eyepop.data.data_types import UserReview
    pa.field("user_review", pa.dictionary(pa.int8(), pa.string())),
]

OBJECT_STRUCT = pa.struct(_object_fields)
OBJECT_SCHEMA = pa.schema(_object_fields)

_class_fields = [
    # Python type is string, there should be only as small number of
    # classLabel values per prediction, hence the dictionary encoding.
    pa.field("classLabel", pa.dictionary(pa.int32(), pa.string())),
    pa.field("confidence", pa.float16()),
    # from eyepop.data.data_types import UserReview
    pa.field("user_review", pa.dictionary(pa.int8(), pa.string())),
]

CLASS_STRUCT = pa.struct(_class_fields)
CLASS_SCHEMA = pa.schema(_class_fields)

_annotation_fields = [
    # from eyepop.data.data_types import AnnotationType
    pa.field("type", pa.dictionary(pa.int8(), pa.string())),
    # from eyepop.data.data_types import AutoAnnotate
    pa.field("source", pa.dictionary(pa.int32(), pa.string())),
    # from eyepop.data.data_types import UserReview
    pa.field("user_review", pa.dictionary(pa.int8(), pa.string())),
    pa.field("objects", pa.list_(OBJECT_STRUCT)),
    pa.field("classes", pa.list_(CLASS_STRUCT))
]

ANNOTATION_STRUCT = pa.struct(_annotation_fields)

ANNOTATION_SCHEMA = pa.schema(_annotation_fields)

_asset_fields = [
    pa.field("uuid", pa.string()),
    pa.field("external_id", pa.string()),
    pa.field("created_at", pa.timestamp("ms")),
    pa.field("updated_at", pa.timestamp("ms")),
    pa.field("asset_url", pa.string()),
    pa.field("original_image_width", pa.uint16()),
    pa.field("original_image_height", pa.uint16()),
    # Python type is string, there should be only as small number of
    # partition values per asset list, hence the dictionary encoding.
    pa.field("partition", pa.dictionary(pa.int32(), pa.string())),
    pa.field("review_priority", pa.float16()),
    pa.field("model_relevance", pa.float16()),
    pa.field("annotations", pa.list_(ANNOTATION_STRUCT)),
]

ASSET_STRUCT = pa.struct(_asset_fields)

ASSET_SCHEMA = pa.schema(_asset_fields)
