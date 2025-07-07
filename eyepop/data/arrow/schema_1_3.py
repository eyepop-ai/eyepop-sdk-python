import pyarrow as pa

from eyepop.data.data_types import MIME_TYPE_APACHE_ARROW_FILE

""" Arrow schema for Asset export/import form Data API. """

MIME_TYPE_APACHE_ARROW_FILE_VERSIONED = f"{MIME_TYPE_APACHE_ARROW_FILE};version=1.2"

# BEGIN: Extension since v1.3
_embedding_fields = [
    pa.field(name="embedding", type=pa.list_(pa.float16())),
    pa.field(name="x", type=pa.float16()),
    pa.field(name="y", type=pa.float16()),
]
EMBEDDING_STRUCT = pa.struct(_embedding_fields)
EMBEDDING_SCHEMA = pa.schema(_embedding_fields)
# END: Extension since v1.3

# BEGIN: Extension since v1.2
_text_fields = [
    pa.field(name="confidence", type=pa.float16()),
    pa.field(name="text", type=pa.string()),
    pa.field(name="category", type=pa.dictionary(pa.int32(), pa.string())),
]
TEXT_STRUCT = pa.struct(_text_fields)
TEXT_SCHEMA = pa.schema(_text_fields)
# END: Extension since v1.2

# BEGIN: Extension since v1.1
_key_point_fields = [
    pa.field(name="classLabel", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="confidence", type=pa.float16()),
    pa.field(name="x", type=pa.float16()),
    pa.field(name="y", type=pa.float16()),
    # optional z coordinate in pixel coordinate system, null="unknown"
    pa.field(name="z", type=pa.float16()),
    #optional flag true=visible, false=invisible, null="unknown"
    pa.field(name="visible", type=pa.bool_()),
    # BEGIN: Extension since v1.2
    pa.field(name="category", type=pa.dictionary(pa.int32(), pa.string())),
    # END: Extension since v1.2
]
KEY_POINT_STRUCT = pa.struct(_key_point_fields)
KEY_POINT_SCHEMA = pa.schema(_key_point_fields)
_key_points_fields = [
    # optional
    pa.field(name="type", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="points", type=pa.list_(KEY_POINT_STRUCT)),
    # BEGIN: Extension since v1.2
    pa.field(name="category", type=pa.dictionary(pa.int32(), pa.string())),
    # END: Extension since v1.2
]
KEY_POINTS_STRUCT = pa.struct(_key_points_fields)
KEY_POINTS_SCHEMA = pa.schema(_key_points_fields)
# END: Extension since v1.1

_object_fields = [
    pa.field(name="classLabel", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="confidence", type=pa.float16()),
    pa.field(name="x", type=pa.float16()),
    pa.field(name="y", type=pa.float16()),
    pa.field(name="width", type=pa.float16()),
    pa.field(name="height", type=pa.float16()),
    # from eyepop.data.data_types import UserReview
    pa.field(name="user_review", type=pa.dictionary(pa.int8(), pa.string())),
    # BEGIN: Extension since v1.1
    pa.field(name="keyPoints", type=pa.list_(KEY_POINTS_STRUCT)),
    # END: Extension since v1.1
    # BEGIN: Extension since v1.2
    pa.field(name="category", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="texts", type=pa.list_(TEXT_STRUCT)),
    # END: Extension since v1.2
]

OBJECT_STRUCT = pa.struct(_object_fields)
OBJECT_SCHEMA = pa.schema(_object_fields)

_class_fields = [
    pa.field(name="classLabel", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="confidence", type=pa.float16()),
    # from eyepop.data.data_types import UserReview
    pa.field(name="user_review", type=pa.dictionary(pa.int8(), pa.string())),
    # BEGIN: Extension since v1.2
    pa.field(name="category", type=pa.dictionary(pa.int32(), pa.string())),
    # END: Extension since v1.2
]

CLASS_STRUCT = pa.struct(_class_fields)
CLASS_SCHEMA = pa.schema(_class_fields)

_annotation_fields = [
    # from eyepop.data.data_types import AnnotationType
    pa.field(name="type", type=pa.dictionary(pa.int8(), pa.string())),
    # from eyepop.data.data_types import AutoAnnotate
    pa.field(name="source", type=pa.dictionary(pa.int32(), pa.string())),
    # from eyepop.data.data_types import UserReview
    pa.field(name="user_review", type=pa.dictionary(pa.int8(), pa.string())),
    pa.field(name="objects", type=pa.list_(OBJECT_STRUCT)),
    pa.field(name="classes", type=pa.list_(CLASS_STRUCT)),
    # read/write, optional, the model that produced this annotation
    pa.field(name="source_model_uuid", type=pa.dictionary(pa.int8(), pa.string())),
    # BEGIN: Extension since v1.1
    pa.field(name="keyPoints", type=pa.list_(KEY_POINTS_STRUCT)),
    # END: Extension since v1.1
    # BEGIN: Extension since v1.2
    pa.field(name="texts", type=pa.list_(TEXT_STRUCT)),
    # END: Extension since v1.2
    # BEGIN: Extension since v1.3
    pa.field(name="embeddings", type=pa.list_(EMBEDDING_STRUCT)),
    # END: Extension since v1.3
]

ANNOTATION_STRUCT = pa.struct(_annotation_fields)

ANNOTATION_SCHEMA = pa.schema(_annotation_fields)

_asset_fields = [
    pa.field(name="uuid", type=pa.string()),
    pa.field(name="external_id", type=pa.string()),
    pa.field(name="created_at", type=pa.timestamp("ms")),
    pa.field(name="updated_at", type=pa.timestamp("ms")),
    pa.field(name="asset_url", type=pa.string()),
    pa.field(name="original_image_width", type=pa.uint16()),
    pa.field(name="original_image_height", type=pa.uint16()),
    pa.field(name="partition", type=pa.dictionary(pa.int32(), pa.string())),
    pa.field(name="review_priority", type=pa.float16()),
    pa.field(name="model_relevance", type=pa.float16()),
    pa.field(name="annotations", type=pa.list_(ANNOTATION_STRUCT)),
]

ASSET_STRUCT = pa.struct(_asset_fields)

ASSET_SCHEMA = pa.schema(_asset_fields)
