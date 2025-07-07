from pyarrow._compute import CastOptions
import pyarrow as pa

from . import schema_1_3 as schema_latest

""" The latest official Arrow schema for the EyePop Dataset API.

The latest officially supported schema is: 1.3

These are references to the types and schemas that are currently 
supported. For backward compatibility, we keep schemas versioned 
and immutable once officially supported.

Using schemas from newer-than-latest version is unstable. 
"""

MIME_TYPE_APACHE_ARROW_FILE_VERSIONED = schema_latest.MIME_TYPE_APACHE_ARROW_FILE_VERSIONED

OBJECT_STRUCT = schema_latest.OBJECT_STRUCT
OBJECT_SCHEMA = schema_latest.OBJECT_SCHEMA

CLASS_STRUCT = schema_latest.CLASS_STRUCT
CLASS_SCHEMA = schema_latest.CLASS_SCHEMA

KEY_POINT_STRUCT = schema_latest.KEY_POINT_STRUCT
KEY_POINT_SCHEMA = schema_latest.KEY_POINT_SCHEMA

KEY_POINTS_STRUCT = schema_latest.KEY_POINTS_STRUCT
KEY_POINTS_SCHEMA = schema_latest.KEY_POINTS_SCHEMA

TEXT_STRUCT = schema_latest.TEXT_STRUCT
TEXT_SCHEMA = schema_latest.TEXT_SCHEMA

EMBEDDING_STRUCT = schema_latest.EMBEDDING_STRUCT
EMBEDDING_SCHEMA = schema_latest.EMBEDDING_SCHEMA

ANNOTATION_STRUCT = schema_latest.ANNOTATION_STRUCT
ANNOTATION_SCHEMA = schema_latest.ANNOTATION_SCHEMA

ASSET_STRUCT = schema_latest.ASSET_STRUCT
ASSET_SCHEMA = schema_latest.ASSET_SCHEMA
