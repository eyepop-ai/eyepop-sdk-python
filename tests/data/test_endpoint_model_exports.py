import io
import json
import unittest
from urllib.parse import unquote_plus

from aioresponses import aioresponses
from yarl import URL

from eyepop import EyePopSdk
from eyepop.data.data_types import (
    ArtifactType,
    ExportedBy,
    ExportedUrlResponse,
    ModelExport,
    ModelExportFormat,
    ModelExportStatus,
    Quantization,
    TargetRuntime,
)
from tests.data.base_endpoint_test import BaseEndpointTest

TEST_MODEL_UUID = 'test_model_uuid'


class TestEndpointModelExports(BaseEndpointTest):

    def setup_token_mock(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

    def requested_urls(self, mock: aioresponses, path: str) -> list[URL]:
        return [key[1] for key in mock.requests.keys() if key[1].path == path]

    def requested_variants(self, url: URL) -> list[str]:
        # yarl versions differ in whether query values are returned percent-decoded.
        return [unquote_plus(v) for v in url.query.getall('variant')]

    @aioresponses()
    def test_upload_model_artifact_default_variant(self, mock: aioresponses):
        self.setup_token_mock(mock)
        put_url = (f'{self.test_data_url}/models/{TEST_MODEL_UUID}/exports/eyepop'
                   f'/formats/{ModelExportFormat.ONNX}/artifacts/model.onnx')
        mock.put(put_url, status=204)

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            endpoint.upload_model_artifact(TEST_MODEL_UUID, ModelExportFormat.ONNX, 'model.onnx',
                                           io.BytesIO(b'test artifact'))

        requested = self.requested_urls(mock, f'/models/{TEST_MODEL_UUID}/exports/eyepop'
                                              f'/formats/{ModelExportFormat.ONNX}/artifacts/model.onnx')
        self.assertEqual(len(requested), 1)
        self.assertEqual(requested[0].query_string, '')

    @aioresponses()
    def test_upload_model_artifact_variant_product(self, mock: aioresponses):
        self.setup_token_mock(mock)
        expected_path = (f'/models/{TEST_MODEL_UUID}/exports/{ExportedBy.qc_ai_hub}'
                         f'/formats/{ModelExportFormat.ONNX}/artifacts/model.onnx')
        put_url = (f'{self.test_data_url}{expected_path}'
                   f'?variant=quantization%3Dint8'
                   f'&variant=target_runtime%3Dcuda_cc_86&variant=target_runtime%3Dcuda_cc_87')
        mock.put(put_url, status=204)

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            endpoint.upload_model_artifact(TEST_MODEL_UUID, ModelExportFormat.ONNX, 'model.onnx',
                                           io.BytesIO(b'test artifact'),
                                           exported_by=ExportedBy.qc_ai_hub,
                                           variant={
                                               'quantization': Quantization.int8,
                                               'target_runtime': [TargetRuntime.cuda_cc_86, TargetRuntime.cuda_cc_87],
                                           })

        requested = self.requested_urls(mock, expected_path)
        self.assertEqual(len(requested), 1)
        self.assertEqual(
            self.requested_variants(requested[0]),
            ['quantization=int8', 'target_runtime=cuda_cc_86', 'target_runtime=cuda_cc_87'],
        )

    @aioresponses()
    def test_export_model_urls_variant(self, mock: aioresponses):
        self.setup_token_mock(mock)
        get_url = (f'{self.test_data_url}/exports/model_urls?model_uuid={TEST_MODEL_UUID}'
                   f'&model_format={ModelExportFormat.ONNX}'
                   f'&variant=quantization%3Dint8&variant=target_runtime%3Dcuda_cc_87')
        mock.get(get_url, status=200, body=json.dumps([ExportedUrlResponse(
            model_uuid=TEST_MODEL_UUID,
            model_format=ModelExportFormat.ONNX,
            exported_url='http://example-storage.test/model.onnx',
        ).model_dump()]))

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            urls = endpoint.export_model_urls([TEST_MODEL_UUID], [ModelExportFormat.ONNX],
                                              variant={
                                                  'quantization': Quantization.int8,
                                                  'target_runtime': TargetRuntime.cuda_cc_87,
                                              })

        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0].exported_url, 'http://example-storage.test/model.onnx')
        requested = self.requested_urls(mock, '/exports/model_urls')
        self.assertEqual(len(requested), 1)
        self.assertEqual(
            self.requested_variants(requested[0]),
            ['quantization=int8', 'target_runtime=cuda_cc_87'],
        )
        self.assertNotIn('device_name', requested[0].query)

    @aioresponses()
    def test_export_model_urls_device_name_deprecation(self, mock: aioresponses):
        self.setup_token_mock(mock)
        get_url = (f'{self.test_data_url}/exports/model_urls?model_uuid={TEST_MODEL_UUID}'
                   f'&model_format={ModelExportFormat.ONNX}&device_name=Test+Device&')
        mock.get(get_url, status=200, body=json.dumps([]))

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            with self.assertWarns(DeprecationWarning):
                endpoint.export_model_urls([TEST_MODEL_UUID], [ModelExportFormat.ONNX], 'Test Device')

        requested = self.requested_urls(mock, '/exports/model_urls')
        self.assertEqual(len(requested), 1)
        self.assertEqual(requested[0].query['device_name'], 'Test Device')

    @aioresponses()
    def test_export_model_artifacts_variant(self, mock: aioresponses):
        self.setup_token_mock(mock)
        get_url = (f'{self.test_data_url}/exports/model_artifacts?model_uuid={TEST_MODEL_UUID}'
                   f'&model_format={ModelExportFormat.ONNX}'
                   f'&artifact_type={ArtifactType.eyepop_bundle}'
                   f'&variant=quantization%3Dint8')
        mock.get(get_url, status=200, body=b'test artifact bytes')

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            stream = endpoint.export_model_artifacts([TEST_MODEL_UUID], [ModelExportFormat.ONNX],
                                                     artifact_type=ArtifactType.eyepop_bundle,
                                                     variant={'quantization': Quantization.int8})
            self.assertEqual(stream.read(), b'test artifact bytes')

        requested = self.requested_urls(mock, '/exports/model_artifacts')
        self.assertEqual(len(requested), 1)
        self.assertEqual(self.requested_variants(requested[0]), ['quantization=int8'])
        self.assertNotIn('device_name', requested[0].query)

    @aioresponses()
    def test_export_model_artifacts_device_name_deprecation(self, mock: aioresponses):
        self.setup_token_mock(mock)
        get_url = (f'{self.test_data_url}/exports/model_artifacts?model_uuid={TEST_MODEL_UUID}'
                   f'&model_format={ModelExportFormat.ONNX}&device_name=Test+Device&')
        mock.get(get_url, status=200, body=b'')

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            with self.assertWarns(DeprecationWarning):
                endpoint.export_model_artifacts([TEST_MODEL_UUID], [ModelExportFormat.ONNX], 'Test Device')

    def test_model_export_parses_variant(self):
        export = ModelExport.model_validate({
            'format': 'ONNX',
            'exported_by': 'eyepop',
            'variant': {'quantization': 'int8', 'target_runtime': 'cuda_cc_87'},
            'status': 'finished',
        })
        self.assertEqual(export.variant, {'quantization': 'int8', 'target_runtime': 'cuda_cc_87'})
        self.assertEqual(export.status, ModelExportStatus.finished)

        export_without_variant = ModelExport.model_validate({
            'format': 'ONNX',
            'exported_by': 'eyepop',
            'status': 'finished',
        })
        self.assertIsNone(export_without_variant.variant)


if __name__ == '__main__':
    unittest.main()
