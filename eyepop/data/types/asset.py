from datetime import datetime
from typing import List, Sequence

from pydantic import BaseModel, ConfigDict

from eyepop.data.types.dataset import AutoAnnotateParams
from eyepop.data.types.enums import AnnotationType, AssetStatus, AutoAnnotate, UserReview
from eyepop.data.types.prediction import Prediction


class AssetAnnotation(BaseModel):
    type: AnnotationType
    user_review: UserReview = UserReview.unknown
    approved_threshold: float | None = None
    auto_annotate: AutoAnnotate | None = None
    source: str | None = None
    predictions: Sequence[Prediction] | None = None
    uncertainty_score: float | None = None


AssetAnnotationResponse = AssetAnnotation


class Asset(BaseModel):
    uuid: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    original_image_width: int | None = None
    original_image_height: int | None = None
    original_duration: float | None = None
    original_frames: int | None = None
    is_transformed: bool | None = None
    status: AssetStatus | None = None
    status_message: str | None = None
    external_id: str | None = None
    partition: str | None = None
    review_priority: float | None = None
    model_relevance: float | None = None
    annotations: List[AssetAnnotation] = []

    ###
    # Denormalized attributes, for convenient serialization/storage detached of a dataset context
    # ###
    dataset_uuid: str | None = None
    account_uuid: str | None = None

    model_config = ConfigDict(
        protected_namespaces=("pydantic_do_not_prevent_model_prefix_",)
    )


AssetResponse = Asset


class AssetAnnotationImport(BaseModel):
    type: AnnotationType
    auto_annotate: AutoAnnotate | None = None
    auto_annotate_params: AutoAnnotateParams | None = None
    predictions: Sequence[Prediction] | None = None
    annotation: Prediction | None = None
    source: str | None = None
    user_review: UserReview | None = None
    approved_threshold: float | None = None
    source_model_uuid: str | None = None


class AssetImport(BaseModel):
    url: str
    mime_type: str | None = None
    file_size_bytes: int | None = None
    external_id: str | None = None
    annotations: list[AssetAnnotationImport] | None = None
    ground_truth: Prediction | None = None
