import pyarrow as pa

from eyepop.data.arrow.pandas.assets import dataframe_from_table, table_from_dataframe
from eyepop.data.arrow.schema import ASSET_SCHEMA as ASSET_SCHEMA_LATEST
from eyepop.data.arrow.schema_0_0 import ASSET_SCHEMA as ASSET_SCHEMA_0_0
from eyepop.data.arrow.schema_1_0 import ASSET_SCHEMA as ASSET_SCHEMA_1_0
from .arrow_test_helpers import create_test_table


def test_arrow_to_pandas():
    _test_arrow_to_pandas(ASSET_SCHEMA_LATEST)

def test_arrow_to_pandas_0_0():
    _test_arrow_to_pandas(ASSET_SCHEMA_0_0)

def test_arrow_to_pandas_1_0():
    _test_arrow_to_pandas(ASSET_SCHEMA_1_0)

def _test_arrow_to_pandas(schema: pa.Schema):
    table_in = create_test_table(schema=schema)
    dataframe = dataframe_from_table(table_in, schema=schema)
    table_out = table_from_dataframe(dataframe, schema=schema)
    assert table_in.schema == table_out.schema
    assert table_in.to_pylist() == table_out.to_pylist()

