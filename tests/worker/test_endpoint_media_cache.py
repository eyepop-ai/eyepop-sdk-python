import json
import time
import warnings
from importlib import resources

import aiohttp
from aioresponses import CallbackResult, aioresponses

import tests
from eyepop import EyePopSdk
from eyepop.worker.worker_types import DEFAULT_PREDICTION_VERSION, Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointMediaCache(BaseEndpointTest):
    """A session's source records to the media cache with `media_cache_seconds` and `media_cache_reencode`.

    Both go in the source body with the instance's names (`api.SourceDef`), and only when set.
    """

    test_source_id = 'test_source_id'
    test_url = 'rtsp://camera.invalid/stream'

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
    def test_load_from_sends_the_recording(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            with warnings.catch_warnings():
                # The SDK's own warnings are UserWarning. Not every warning: aioresponses raises a
                # DeprecationWarning on Python 3.14 (asyncio.iscoroutinefunction) on each request.
                warnings.simplefilter("error", UserWarning)
                endpoint.load_from(self.test_url, media_cache_seconds=600, media_cache_reencode=True).predict()

            self._assert_body(mock, {
                'sourceType': 'URL',
                'url': self.test_url,
                'version': DEFAULT_PREDICTION_VERSION,
                'mediaCacheSeconds': 600,
                'mediaCacheReencode': True,
            })

    @aioresponses()
    def test_load_from_omits_the_recording_when_unset(self, mock: aioresponses):
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
    async def test_upload_warns_at_submit_that_reencode_has_no_effect(self, mock: aioresponses):
        # A session's upload routes take mediaCacheSeconds but not mediaCacheReencode: the call that
        # asks for it is warned, once, where it is made, and the upload goes on without it.
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            mock.post(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}'
                      f'/source?mode=queue&processing=sync&version={DEFAULT_PREDICTION_VERSION}',
                      body=json.dumps({'source_id': self.test_source_id, 'seconds': 0,
                                       'system_timestamp': time.time() * 1000 * 1000 * 1000}))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                job = await endpoint.upload(resources.files(tests) / 'test.jpg', media_cache_reencode=True)
                submitted = [w for w in caught if issubclass(w.category, UserWarning)]
                self.assertIsNotNone(await job.predict())
            self.assertEqual([str(w.message) for w in submitted], [
                'media_cache_reencode is not taken by a session for an upload, and has no effect; '
                'it applies to a url or an asset'])
            self.assertEqual(submitted[0].filename, __file__)
            self.assertEqual(len([w for w in caught if issubclass(w.category, UserWarning)]), 1)
