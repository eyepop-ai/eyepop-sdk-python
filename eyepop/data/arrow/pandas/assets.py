import pyarrow as pa
import pandas as pd
from pyarrow import ListType, StructType

from eyepop.data.arrow.schema import ASSET_SCHEMA
from eyepop.data.arrow.schema_version_conversion import convert


def _pandas_compatible_data_type(data_type: pa.DataType) -> pa.DataType:
    if data_type == pa.float16():
        return pa.float64()
    elif isinstance(data_type, ListType):
        return pa.list_(_pandas_compatible_data_type(data_type.field(0).type))
    elif isinstance(data_type, StructType):
        pandas_compatible_fields = [None] * data_type.num_fields
        for i in range(data_type.num_fields):
            field = data_type.field(i)
            pandas_compatible_field = pa.field(field.name, _pandas_compatible_data_type(field.type))
            pandas_compatible_fields[i] = pandas_compatible_field
        return pa.struct(pandas_compatible_fields)
    else:
        return data_type

def _pandas_compatible_schema(schema: pa.Schema) -> pa.Schema:
    pandas_compatible_fields = [None] * len(schema)
    for i in range(len(schema)):
        field = schema.field(i)
        pandas_compatible_fields[i] = pa.field(field.name, _pandas_compatible_data_type(field.type))
    return pa.schema(fields=pandas_compatible_fields)

PANDAS_COMPAT_SCHEMA = _pandas_compatible_schema(ASSET_SCHEMA)

def table_from_dataframe(df: pd.DataFrame, schema: pa.Schema = ASSET_SCHEMA) -> pa.Table:
    """ Safely converts a pandas dataframe to a pyarrow.Table that complies with the ASSET_SCHEMA.
    """
    compat_schema = PANDAS_COMPAT_SCHEMA
    if schema != ASSET_SCHEMA:
        compat_schema = _pandas_compatible_schema(schema)
    unsafe_table = pa.Table.from_pandas(df, schema=compat_schema)
    return unsafe_table.cast(schema)


def dataframe_from_table(table: pa.Table, schema: pa.Schema = ASSET_SCHEMA) -> pd.DataFrame:
    """ Safely converts a pyarrow.Table with ASSET_SCHEMA to a pandas dataframe.
    """
    type_mapping = {
        # Needed that to avoid exception when categorical column is completely None
        pa.dictionary(pa.int32(), pa.string()): pd.StringDtype(),
    }
    return convert(table, schema).to_pandas(types_mapper=type_mapping.get)
