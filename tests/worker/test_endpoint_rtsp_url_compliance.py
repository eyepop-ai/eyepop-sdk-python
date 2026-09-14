import json
import time

import aiohttp
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.worker_types import DEFAULT_PREDICTION_VERSION, Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointRtspUrlCompliance(BaseEndpointTest):
    """`rtsp_force_non_compliant_url` reaches the worker, and only when asked for.

    The worker builds RFC 2326 compliant SETUP URLs by default, which is what a
    camera advertising an absolute control URL needs. The option exists for the
    servers that require the older construction, so the value a caller does not
    set must stay off the wire entirely rather than be sent as false - the
    worker's default is what should decide, not the SDK's.
    """

    test_source_id = 'test_source_id'
    test_url = 'rtsp://camera.invalid/axis-media/media.amp'

    def _setup_worker(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                 body=json.dumps({'pop': Pop(components=[]).model_dump()}))

    def _mock_source(self, mock: aioresponses):
        def load_from(url, **kwargs) -> CallbackResult:
            return CallbackResult(status=200, body=json.dumps(
                {'source_id': self.test_source_id, 'seconds': 0,
                 'system_timestamp': time.time() * 1000 * 1000 * 1000}))

        mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                   callback=load_from)

    def _assert_body(self, mock: aioresponses, expected: dict):
        mock.assert_called_with(
            f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
            method='PATCH',
            headers={
                'Accept': 'application/jsonl',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.test_access_token}'
            },
            data=json.dumps(expected),
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    def test_load_from_omits_the_option_when_unset(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url).predict()

            self._assert_body(mock, {
                'sourceType': 'URL',
                'url': self.test_url,
                'version': DEFAULT_PREDICTION_VERSION,
            })

    @aioresponses()
    def test_load_from_sends_the_option_when_requested(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url, rtsp_force_non_compliant_url=True).predict()

            self._assert_body(mock, {
                'sourceType': 'URL',
                'url': self.test_url,
                'version': DEFAULT_PREDICTION_VERSION,
                'rtspForceNonCompliantUrl': True,
            })

    @aioresponses()
    def test_load_from_sends_an_explicit_false(self, mock: aioresponses):
        """Explicit false is not the same as unset.

        It pins the compliant construction against a worker whose default might
        differ.
        """
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url, rtsp_force_non_compliant_url=False).predict()

            self._assert_body(mock, {
                'sourceType': 'URL',
                'url': self.test_url,
                'version': DEFAULT_PREDICTION_VERSION,
                'rtspForceNonCompliantUrl': False,
            })
