import json
from importlib import resources

import pytest
from eyepop.data.data_types import Prediction, AnnotationType, AssetAnnotationResponse, UserReview

from eyepop.data.arrow.eyepop.assets import eyepop_assets_from_table, table_from_eyepop_assets
from eyepop.data.arrow.schema_0_0 import ASSET_SCHEMA as ASSET_SCHEMA_0_0
from eyepop.data.arrow.schema_1_0 import ASSET_SCHEMA as ASSET_SCHEMA_1_0
from eyepop.data.arrow.schema_1_1 import ASSET_SCHEMA as ASSET_SCHEMA_1_1
from eyepop.data.arrow.schema_1_2 import ASSET_SCHEMA as ASSET_SCHEMA_1_2
from eyepop.data.arrow.schema_1_3 import ASSET_SCHEMA as ASSET_SCHEMA_1_3

from eyepop.data.arrow.eyepop.annotations import table_from_eyepop_annotations, eyepop_annotations_from_table
from eyepop.data.data_normalize import normalize_eyepop_annotations, normalize_eyepop_prediction
from . import files
from .arrow_test_helpers import create_test_table


class TestArrowToFromAnnotation:
    @pytest.mark.parametrize("file_name, n", [
        ("prediction_0_bbox.json", 1),
        ("prediction_1_bbox.json", 2),
        ("prediction_2_bbox.json", 3),
        ("prediction_3_classes.json", 4),
        ("prediction_4_bbox_and_classes.json", 5),
        ("prediction_2_keypoints_2_objects.json", 6),
    ])
    def test_prediction_from_file(self, file_name, n):
        test_json = resources.files(files) / file_name
        with test_json.open("r") as f:
            source_prediction = Prediction(**json.load(f))
        normalize_eyepop_prediction(source_prediction)

        source_annotations = [
            AssetAnnotationResponse(
                type=AnnotationType.ground_truth,
                user_review=UserReview.unknown,
                source="foo bar",
                annotation=source_prediction
            )
        ]
        table = table_from_eyepop_annotations(source_annotations)
        assert table is not None
        assert table.num_rows == len(source_annotations)
        target_annotations = eyepop_annotations_from_table(table)
        assert target_annotations is not None
        normalize_eyepop_annotations(target_annotations)
        if source_annotations != target_annotations:
            assert source_annotations == target_annotations

    def test_0_0(self):
        """ verifying that the new field `source_model_uuid` does not confuse 0.0 """
        source_table = create_test_table(schema=ASSET_SCHEMA_0_0)
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_0_0)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_0_0)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

    def test_0_0_to_1_0(self):
        source_table = create_test_table(schema=ASSET_SCHEMA_0_0)
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_0_0)
        assert target_assets is not None
        assert len(target_assets) == 1
        assert len(target_assets[0].annotations) == 2
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_0)
        assert target_table is not None
        assert target_table.schema == ASSET_SCHEMA_1_0
        for column_name in ASSET_SCHEMA_0_0.names:
            if column_name != "annotations":
                assert target_table.column(column_name) == source_table.column(column_name)

    def test_1_0_to_0_0(self):
        source_table = create_test_table(schema=ASSET_SCHEMA_1_0)
        target_assets = eyepop_assets_from_table(source_table)
        assert target_assets is not None
        assert len(target_assets) == 1
        assert len(target_assets[0].annotations) == 2
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_0_0)
        assert target_table is not None
        assert target_table.schema == ASSET_SCHEMA_0_0
        for column_name in ASSET_SCHEMA_0_0.names:
            if column_name != "annotations":
                assert target_table.column(column_name) == source_table.column(column_name)

    def test_1_0(self):
        """ verifying that the new field `source_model_uuid` in 1.0 are converted """
        source_table = create_test_table(schema=ASSET_SCHEMA_1_0)
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_0)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_0)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

    def test_1_1(self):
        """ verifying that the new field `key_points` in 1.1 are converted """
        source_table = create_test_table(schema=ASSET_SCHEMA_1_1, test_file_name="prediction_2_keypoints.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_1)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_1)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

        source_table = create_test_table(schema=ASSET_SCHEMA_1_1, test_file_name="prediction_2_keypoints_2_objects.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_1)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_1)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

    def test_1_1_to_0_0(self):
        source_table = create_test_table(schema=ASSET_SCHEMA_1_1, test_file_name="prediction_2_keypoints_2_objects.json")
        target_assets = eyepop_assets_from_table(source_table)
        assert target_assets is not None
        assert len(target_assets) == 1
        assert len(target_assets[0].annotations) == 2
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_0)
        assert target_table is not None
        assert target_table.schema == ASSET_SCHEMA_1_0
        for column_name in ASSET_SCHEMA_1_0.names:
            if column_name != "annotations":
                assert target_table.column(column_name) == source_table.column(column_name)

    def test_1_2(self):
        """ verifying that the new fields `text` and 'category' in 1.2 are converted """
        source_table = create_test_table(schema=ASSET_SCHEMA_1_2, test_file_name="prediction_2_keypoints_with_category.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_2)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_2)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

        source_table = create_test_table(schema=ASSET_SCHEMA_1_2, test_file_name="prediction_2_objects_category_texts.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_2)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_2)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

    def test_1_2_to_1_1(self):
        source_table = create_test_table(schema=ASSET_SCHEMA_1_2, test_file_name="prediction_2_objects_category_texts.json")
        target_assets = eyepop_assets_from_table(source_table)
        assert target_assets is not None
        assert len(target_assets) == 1
        assert len(target_assets[0].annotations) == 2
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_1)
        assert target_table is not None
        assert target_table.schema == ASSET_SCHEMA_1_1
        for column_name in ASSET_SCHEMA_1_0.names:
            if column_name != "annotations":
                assert target_table.column(column_name) == source_table.column(column_name)

    def test_1_3(self):
        """ verify that the new field `embeddings` in 1.3 are converted """
        source_table = create_test_table(schema=ASSET_SCHEMA_1_3, test_file_name="prediction_1_embeddings.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_3)
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_3)
        assert target_table.schema == source_table.schema
        assert target_table == source_table

    def test_1_3_to_1_2(self):
        source_table = create_test_table(schema=ASSET_SCHEMA_1_3, test_file_name="prediction_1_embeddings.json")
        target_assets = eyepop_assets_from_table(source_table)
        assert target_assets is not None
        assert len(target_assets) == 1
        assert len(target_assets[0].annotations) == 2
        target_table = table_from_eyepop_assets(target_assets, schema=ASSET_SCHEMA_1_2)
        assert target_table is not None
        assert target_table.schema == ASSET_SCHEMA_1_2
        for column_name in ASSET_SCHEMA_1_2.names:
            if column_name != "annotations":
                assert target_table.column(column_name) == source_table.column(column_name)

    def test_assets(self):
        """ verify that the new denormalized fields account_uuid and dataset_uuids are being filled """
        source_table = create_test_table(schema=ASSET_SCHEMA_1_3, test_file_name="prediction_1_embeddings.json")
        target_assets = eyepop_assets_from_table(source_table, schema=ASSET_SCHEMA_1_3)
        assert len(target_assets) > 0
        for target_asset in target_assets:
            assert target_asset.dataset_uuid is None
            assert target_asset.account_uuid is None
        target_assets = eyepop_assets_from_table(
            source_table,
            schema=ASSET_SCHEMA_1_3,
            dataset_uuid="test_dataset_uuid",
            account_uuid="test_account_uuid",
        )
        assert len(target_assets) > 0
        for target_asset in target_assets:
            assert target_asset.dataset_uuid == "test_dataset_uuid"
            assert target_asset.account_uuid == "test_account_uuid"
