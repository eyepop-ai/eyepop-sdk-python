from typing import cast

import pyarrow as pa
from eyepop.data.data_types import Asset

from eyepop.data.arrow.eyepop.annotations import table_from_eyepop_annotations, eyepop_annotations_from_pylist
from eyepop.data.arrow.schema import ASSET_SCHEMA
from eyepop.data.arrow.schema_version_conversion import convert
from eyepop.data.data_normalize import CONFIDENCE_N_DIGITS

UNKNOWN_MIME_TYPE = "application/octet-stream"


def table_from_eyepop_assets(assets: list[Asset], schema: pa.Schema = ASSET_SCHEMA) -> pa.Table:
    return pa.Table.from_batches(batches=[record_batch_from_eyepop_assets(assets, schema)], schema=schema)


def record_batch_from_eyepop_assets(assets: list[Asset], schema: pa.Schema = ASSET_SCHEMA) -> pa.RecordBatch:
    uuids: list[any] = [None] * len(assets)
    external_ids = uuids.copy()
    created_ats = uuids.copy()
    updated_ats = uuids.copy()
    asset_urls = uuids.copy()
    original_image_widths = uuids.copy()
    original_image_height = uuids.copy()
    partitions = uuids.copy()
    review_priorities = uuids.copy()
    model_relevance = uuids.copy()
    annotationss = uuids.copy()

    for i, e in enumerate(assets):
        uuids[i] = e.uuid
        external_ids[i] = e.external_id
        created_ats[i] = e.created_at
        updated_ats[i] = e.updated_at
        asset_urls[i] = None
        original_image_widths[i] = None
        original_image_height[i] = None
        partitions[i] = e.partition
        review_priorities[i] = e.review_priority
        model_relevance[i] = e.model_relevance
        if e.annotations is None:
            annotationss[i] = None
        else:
            annotation_fields = cast(pa.StructType, cast(pa.ListType, schema.field("annotations").type).value_type).fields
            annotations_schema = pa.schema(annotation_fields)
            if len(e.annotations) > 0:
                annotationss[i] = table_from_eyepop_annotations(e.annotations, schema=annotations_schema).to_struct_array()
            else:
                annotationss[i] = pa.chunked_array([], type=pa.struct(annotation_fields))

    return pa.RecordBatch.from_arrays(
        [
            pa.array(uuids),
            pa.array(external_ids),
            pa.array(created_ats),
            pa.array(updated_ats),
            pa.array(asset_urls),
            pa.array(original_image_widths),
            pa.array(original_image_height),
            pa.array(partitions).dictionary_encode(),
            pa.array(review_priorities),
            pa.array(model_relevance),
            annotationss
        ], schema=schema
    )


def eyepop_assets_from_table(
        table: pa.Table,
        schema: pa.Schema = ASSET_SCHEMA,
        dataset_uuid: str | None = None,
        account_uuid: str | None = None,
) -> list[Asset]:
    table = convert(table, schema)
    assets = [None] * table.num_rows
    i = 0
    for batch in table.to_reader():
        uuids = batch.column(0).to_pylist()
        external_ids = batch.column(1).to_pylist()
        created_ats = batch.column(2).to_pylist()
        updated_ats = batch.column(3).to_pylist()
        asset_urls = batch.column(4).to_pylist()
        original_image_widths = batch.column(5).to_pylist()
        original_image_height = batch.column(6).to_pylist()
        partitions = batch.column(7).to_pylist()
        review_priorities = batch.column(8).to_pylist()
        model_relevances = batch.column(9).to_pylist()
        annotationss = batch.column(10).to_pylist()

        for j in range(len(uuids)):
            review_priority = review_priorities[j]
            if review_priority is not None:
                review_priority = round(review_priority, CONFIDENCE_N_DIGITS)
            model_relevance = model_relevances[j]
            if model_relevance is not None:
                model_relevance = round(model_relevance, CONFIDENCE_N_DIGITS)
            assets[i] = Asset(
                uuid=uuids[j],
                mime_type=UNKNOWN_MIME_TYPE,
                external_id=external_ids[j],
                created_at=created_ats[j],
                updated_at=updated_ats[j],
                original_image_width=original_image_widths[j],
                original_image_height=original_image_height[j],
                partition=partitions[j],
                review_priority=review_priority,
                model_relevance=model_relevance,
                annotations=eyepop_annotations_from_pylist(annotationss[j]) if annotationss[j] is not None else None,
                dataset_uuid=dataset_uuid,
                account_uuid=account_uuid,
            )
            i += 1
    return assets
