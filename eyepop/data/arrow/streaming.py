import typing
from io import BytesIO

import pyarrow as pa
from pyarrow.ipc import IpcWriteOptions

class RestartAbleArrowStream(typing.AsyncIterable[bytes]):
    table: pa.Table
    schema: pa.Schema
    max_chunk_size: int
    callback: typing.Callable[[int], None] | None
    def __init__(self,
                 table: pa.Table,
                 schema: pa.Schema | None,
                 max_chunk_size: int,
                 callback: typing.Callable[[int], None] | None = None):
        self.table = table
        self.schema = schema if schema is not None else table.schema
        self.max_chunk_size = max_chunk_size
        self.callback = callback

    def __aiter__(self) -> typing.AsyncIterator[bytes]:
        return _stream(self.table, self.schema, self.max_chunk_size, self.callback).__aiter__()

async def _stream(
        table: pa.Table,
        schema: pa.Schema,
        max_chunk_size: int,
        callback: typing.Callable[[int], None] | None
) -> typing.AsyncIterable[bytes]:
    buffer = BytesIO()
    writer = pa.ipc.new_stream(
        buffer,
        schema=schema,
        options=IpcWriteOptions(emit_dictionary_deltas=True))
    for batch in table.to_batches(max_chunksize=max_chunk_size):
        writer.write_batch(batch)
        chunk = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        yield chunk
        if callback is not None:
            callback(batch.num_rows)
    writer.close()

def stream_arrow_table(
        table: pa.Table,
        schema: pa.Schema | None = None,
        max_chunk_size: int = 1024,
        callback: typing.Callable[[int], None] = None,
) -> typing.AsyncIterable[bytes]:
    return RestartAbleArrowStream(table, schema, max_chunk_size, callback)