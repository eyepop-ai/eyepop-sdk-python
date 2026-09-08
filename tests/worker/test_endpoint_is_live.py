import inspect
import json
from importlib import resources

import pytest
from aioresponses import CallbackResult, aioresponses

import tests
from eyepop import EyePopSdk
from eyepop.worker.worker_syncify import SyncWorkerEndpoint
from eyepop.worker.worker_types import Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointIsLive(BaseEndpointTest):
    """is_live and captured_at_offset_ns have to reach the wire from both endpoints.

    Both endpoints declared the parameters long before they forwarded them, so
    these assert the query string rather than the call succeeding. Both are
    keyword-only on upload() and the sync upload_stream(), which published
    their positional order without them; test_positional_order_preserved
    guards that.
    """
    test_source_id = 'test_source_id'
    test_file = str(resources.files(tests) / 'test.jpg')
    test_offset_ns = 1234567890

    def setup_pop_mock(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}', status=200,
                 body=json.dumps({'pop': Pop(components=[]).model_dump()}))

    def mock_source_post(self, mock: aioresponses, seen: list[str]):
        def on_post(url, **kwargs) -> CallbackResult:
            seen.append(str(url))
            return CallbackResult(status=200, body=json.dumps(
                {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': 0}))

        mock.post(
            f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}'
            f'/source?mode=queue&processing=sync&isLive=True'
            f'&capturedAtOffsetNs={self.test_offset_ns}&version=2',
            callback=on_post)

    @aioresponses()
    def test_sync_upload_stream_sends_is_live(self, mock: aioresponses):
        self.setup_pop_mock(mock)
        seen: list[str] = []
        self.mock_source_post(mock, seen)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            with open(self.test_file, 'rb') as file:
                job = endpoint.upload_stream(
                    file, 'image/jpeg', is_live=True,
                    captured_at_offset_ns=self.test_offset_ns)
                job.predict()
        self.assertEqual(len(seen), 1)
        self.assertIn('isLive=True', seen[0])
        self.assertIn(f'capturedAtOffsetNs={self.test_offset_ns}', seen[0])

    @aioresponses()
    def test_sync_upload_sends_is_live(self, mock: aioresponses):
        self.setup_pop_mock(mock)
        seen: list[str] = []
        self.mock_source_post(mock, seen)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            job = endpoint.upload(
                self.test_file, is_live=True,
                captured_at_offset_ns=self.test_offset_ns)
            job.predict()
        self.assertEqual(len(seen), 1)
        self.assertIn('isLive=True', seen[0])
        self.assertIn(f'capturedAtOffsetNs={self.test_offset_ns}', seen[0])

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_upload_sends_is_live(self, mock: aioresponses):
        self.setup_pop_mock(mock)
        seen: list[str] = []
        self.mock_source_post(mock, seen)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            job = await endpoint.upload(
                self.test_file, is_live=True,
                captured_at_offset_ns=self.test_offset_ns)
            await job.predict()
        self.assertEqual(len(seen), 1)
        self.assertIn('isLive=True', seen[0])
        self.assertIn(f'capturedAtOffsetNs={self.test_offset_ns}', seen[0])

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_upload_stream_sends_is_live(self, mock: aioresponses):
        self.setup_pop_mock(mock)
        seen: list[str] = []
        self.mock_source_post(mock, seen)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            with open(self.test_file, 'rb') as file:
                job = await endpoint.upload_stream(
                    file, 'image/jpeg', is_live=True,
                    captured_at_offset_ns=self.test_offset_ns)
                await job.predict()
        self.assertEqual(len(seen), 1)
        self.assertIn('isLive=True', seen[0])
        self.assertIn(f'capturedAtOffsetNs={self.test_offset_ns}', seen[0])

    def test_positional_order_preserved(self):
        """The published positional order must survive adding these parameters.

        A caller passing params positionally would otherwise bind it to
        is_live, dropping the component params and sending a list as isLive.
        """
        from eyepop.worker.worker_endpoint import WorkerEndpoint

        for func, expected in [
            (WorkerEndpoint.upload,
             ['self', 'location', 'video_mode', 'params']),
            (SyncWorkerEndpoint.upload_stream,
             ['self', 'stream', 'mime_type', 'video_mode', 'params']),
        ]:
            spec = inspect.getfullargspec(func)
            self.assertEqual(spec.args[:len(expected)], expected)
            for name in ('is_live', 'captured_at_offset_ns'):
                self.assertIn(name, spec.kwonlyargs)
                self.assertNotIn(name, spec.args)
