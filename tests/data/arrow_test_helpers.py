import json
from datetime import datetime, timezone
from importlib import resources
from uuid import uuid4

import pyarrow as pa
from eyepop.data.data_types import Prediction, AssetResponse, AssetStatus, AssetAnnotationResponse, AnnotationType, \
    UserReview

from eyepop.data.arrow.eyepop.assets import table_from_eyepop_assets
from eyepop.data.data_normalize import normalize_eyepop_prediction
from eyepop.data.arrow.schema import ASSET_SCHEMA as ASSET_SCHEMA_LATEST
from tests.data import files


def create_test_table(schema: pa.Schema = ASSET_SCHEMA_LATEST, test_file_name: str = "prediction_2_bbox.json"):
    test_json = resources.files(files) / test_file_name
    with test_json.open("r") as f:
        source_prediction = Prediction(**json.load(f))
    normalize_eyepop_prediction(source_prediction)
    source_assets = [
        AssetResponse(
            uuid=uuid4().hex,
            mime_type="image/jpeg",
            status=AssetStatus.accepted,
            created_at=datetime(2000, 1, 1, 0, 0, 0, 0, timezone.utc),
            updated_at=datetime(2000, 1, 1, 0, 0, 0, 0, timezone.utc),
            annotations=[
                AssetAnnotationResponse(
                    type=AnnotationType.ground_truth,
                    user_review=UserReview.unknown,
                    source="foo bar",
                    annotation=source_prediction
                ),
                AssetAnnotationResponse(
                    type=AnnotationType.auto,
                    user_review=UserReview.unknown,
                    source="magic",
                    annotation=source_prediction,
                    source_model_uuid="magic uuid"
                )
            ]
        )
    ]
    table_in = table_from_eyepop_assets(source_assets, schema)
    return table_in
