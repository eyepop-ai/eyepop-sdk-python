import pyarrow as pa


def convert(source_table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    if source_table.schema == target_schema:
        return source_table
    source_names = set(source_table.schema.names)
    target_columns = []
    for name in target_schema.names:
        target_field = target_schema.field(name)
        if name in source_names:
            source_column = source_table.column(name)
            source_field = source_table.field(name)
            if pa.types.is_list(source_field.type):
                if not pa.types.is_list(target_field.type):
                    raise ValueError(f"field {name} is list type in source schema but not in target schema")
                target_column_schema = pa.schema(target_field.type.value_type)
                target_column = pa.chunked_array([source_column.to_pylist()], target_field.type)
            else:
                target_column = source_column.cast(target_field.type)
        else:
            if not target_field.nullable:
                raise ValueError(f"field {name} of target schema is not nullable and does not exist in source table")
            target_column = pa.chunked_array([], target_field.type)
        target_columns.append(target_column)
    target_table = pa.Table.from_arrays(target_columns, schema=target_schema)
    return target_table


