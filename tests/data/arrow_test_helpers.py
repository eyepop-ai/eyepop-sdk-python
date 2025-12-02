import json
from datetime import datetime, timezone
from importlib import resources
from typing import Sequence
from uuid import uuid4

import pyarrow as pa

from eyepop.data.arrow.eyepop.assets import table_from_eyepop_assets
from eyepop.data.arrow.schema import ASSET_SCHEMA as ASSET_SCHEMA_LATEST
from eyepop.data.data_normalize import normalize_eyepop_prediction
from eyepop.data.data_types import (
    AnnotationType,
    AssetAnnotationResponse,
    AssetResponse,
    AssetStatus,
    Prediction,
    UserReview,
)
from tests.data import files


def create_test_table(schema: pa.Schema = ASSET_SCHEMA_LATEST, test_files: str | Sequence[str] = "prediction_2_bbox.json"):
    if isinstance(test_files, str):
        test_json = resources.files(files) / test_files
        with test_json.open("r") as f:
            source_prediction = Prediction(**json.load(f))
            normalize_eyepop_prediction(source_prediction)
            source_predictions = (source_prediction,)
    elif isinstance(test_files, Sequence):
        source_predictions = []
        for test_file in test_files:
            test_json = resources.files(files) / test_file
            with test_json.open("r") as f:
                source_prediction = Prediction(**json.load(f))
                normalize_eyepop_prediction(source_prediction)
                source_predictions.append(source_prediction)
    else:
        raise ValueError("test_files must be a string or a sequence of strings")

    source_assets = [
        AssetResponse(
            uuid=uuid4().hex,
            mime_type="image/jpeg",
            original_duration=10.0,
            original_frames=4711,
            original_image_width=640,
            original_image_height=480,
            status=AssetStatus.accepted,
            created_at=datetime(2000, 1, 1, 0, 0, 0, 0, timezone.utc),
            updated_at=datetime(2000, 1, 1, 0, 0, 0, 0, timezone.utc),
            annotations=[
                AssetAnnotationResponse(
                    type=AnnotationType.ground_truth,
                    user_review=UserReview.unknown,
                    source="foo bar",
                    predictions=source_predictions,
                    annotation=source_predictions[0]
                ),
                AssetAnnotationResponse(
                    type=AnnotationType.auto,
                    user_review=UserReview.unknown,
                    source="magic",
                    predictions=source_predictions,
                    annotation=source_predictions[0],
                    source_model_uuid="magic uuid"
                )
            ]
        )
    ]
    table_in = table_from_eyepop_assets(source_assets, schema)
    return table_in
