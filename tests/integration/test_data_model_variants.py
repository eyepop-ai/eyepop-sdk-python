"""Integration tests for model artifact variants (OPA-75 / OPA-69).

Requires a staging EYEPOP_URL with EYEPOP_API_KEY and EYEPOP_ACCOUNT_ID; creates a
model on the integration account, uploads variant artifacts, and deletes the model.
"""
import io
import json
import os
import uuid

import aiohttp
import pytest

from eyepop import EyePopSdk
from eyepop.data.data_types import (
    ModelCreate,
    ModelExportFormat,
    ModelExportStatus,
    Quantization,
    TargetRuntime,
)

TEST_ARTIFACT = b'not a real onnx model, integration test payload'
TEST_MODEL_JSON = json.dumps({"test": "integration test model manifest"}).encode()

VARIANT_CC_86 = {'quantization': str(Quantization.int8), 'target_runtime': str(TargetRuntime.cuda_cc_86)}
VARIANT_CC_87 = {'quantization': str(Quantization.int8), 'target_runtime': str(TargetRuntime.cuda_cc_87)}


def requires_data_api():
    return pytest.mark.skipif(
        not os.getenv("EYEPOP_API_KEY") or not os.getenv("EYEPOP_ACCOUNT_ID"),
        reason="EYEPOP_API_KEY and EYEPOP_ACCOUNT_ID environment variables not set",
    )


@requires_data_api()
def test_model_artifact_variant_lifecycle():
    with EyePopSdk.dataEndpoint() as endpoint:
        endpoint.connect()
        model = endpoint.create_model(ModelCreate(
            name=f'sdk-integration-variants-{uuid.uuid4().hex[:8]}',
            description='eyepop-sdk-python integration test for model artifact variants, safe to delete',
        ))
        try:
            # Default variant: artifact + model.json commit.
            endpoint.upload_model_artifact(
                model.uuid, ModelExportFormat.ONNX, 'model.onnx', io.BytesIO(TEST_ARTIFACT))
            endpoint.upload_model_artifact(
                model.uuid, ModelExportFormat.ONNX, 'model.json', io.BytesIO(TEST_MODEL_JSON))

            # One upload registers the binary for the cartesian product int8 x {cc_86, cc_87};
            # the model.json commit with the identical variant set finishes both rows.
            product_variant = {
                'quantization': Quantization.int8,
                'target_runtime': [TargetRuntime.cuda_cc_86, TargetRuntime.cuda_cc_87],
            }
            endpoint.upload_model_artifact(
                model.uuid, ModelExportFormat.ONNX, 'model.onnx', io.BytesIO(TEST_ARTIFACT),
                variant=product_variant)
            endpoint.upload_model_artifact(
                model.uuid, ModelExportFormat.ONNX, 'model.json', io.BytesIO(TEST_MODEL_JSON),
                variant=product_variant)

            # The model response includes exports only for available/published models.
            endpoint.publish_model(model.uuid)

            # All three export rows are registered and finished.
            model = endpoint.get_model(model.uuid)
            onnx_exports = [e for e in model.exports or [] if e.format == ModelExportFormat.ONNX]
            assert len(onnx_exports) == 3
            assert all(e.status == ModelExportStatus.finished for e in onnx_exports)
            variants = [e.variant or {} for e in onnx_exports]
            assert {} in variants
            assert VARIANT_CC_86 in variants
            assert VARIANT_CC_87 in variants

            # Exact variant match.
            urls = endpoint.export_model_urls([model.uuid], [ModelExportFormat.ONNX],
                                              variant=VARIANT_CC_87)
            assert len(urls) == 1
            assert urls[0].model_uuid == model.uuid

            # Non-matching variant falls back to the default variant.
            urls = endpoint.export_model_urls([model.uuid], [ModelExportFormat.ONNX],
                                              variant={'quantization': str(Quantization.fp16)})
            assert len(urls) == 1

            # No variant params resolves the default variant.
            urls = endpoint.export_model_urls([model.uuid], [ModelExportFormat.ONNX])
            assert len(urls) == 1

            # Artifact bytes round-trip for an exact variant match.
            stream = endpoint.export_model_artifacts([model.uuid], [ModelExportFormat.ONNX],
                                                     variant=VARIANT_CC_86)
            assert len(stream.read()) > 0

            # Finished exports are immutable: re-upload into a finished variant is rejected.
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                endpoint.upload_model_artifact(
                    model.uuid, ModelExportFormat.ONNX, 'model.onnx', io.BytesIO(TEST_ARTIFACT),
                    variant=VARIANT_CC_87)
            assert exc_info.value.status == 409
        finally:
            endpoint.delete_model(model.uuid)
