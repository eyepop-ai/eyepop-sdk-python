import io
import json
import time
import unittest

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses
from pydantic import ValidationError

from eyepop import EyePopSdk
from eyepop.worker.pipeline_backend import _set_source_command
from eyepop.worker.worker_jobs import WorkerSource
from eyepop.worker.worker_types import DEFAULT_PREDICTION_VERSION, Pop, TargetResolution
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestTargetResolutionModel(unittest.TestCase):
    """A target box takes whole-number sides from 64 to 16384 pixels."""

    def test_accepts_the_bounds(self):
        self.assertEqual(TargetResolution(width=64, height=16384).model_dump(), {'width': 64, 'height': 16384})

    def test_refuses_sides_out_of_range_or_not_whole(self):
        for width, height in [(63, 720), (1280, 16385), (0, 720), (1280.5, 720)]:
            with self.subTest(width=width, height=height):
                with pytest.raises(ValidationError):
                    TargetResolution(width=width, height=height)


class TestEndpointTargetResolution(BaseEndpointTest):
    """A source's targetResolution reaches the worker on the routes that start a source."""

    test_source_id = 'test_source_id'
    test_url = 'http://examle-media.test/test.png'

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

    def _sent_body(self, mock: aioresponses) -> dict:
        calls = [call for key, calls in mock.requests.items()
                 if key[0] == 'PATCH' and str(key[1]).split('?')[0].endswith('/source') for call in calls]
        self.assertTrue(calls)
        return json.loads(calls[-1].kwargs['data'])

    @aioresponses()
    def test_load_from_sends_the_target_resolution(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url, target_resolution=TargetResolution(width=1280, height=720)).predict()

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=json.dumps({
                    'sourceType': 'URL',
                    'url': self.test_url,
                    'version': DEFAULT_PREDICTION_VERSION,
                    'targetResolution': {'width': 1280, 'height': 720},
                }),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    def test_a_group_sends_the_target_resolution_once(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from_group([self.test_url, self.test_url],
                                     target_resolution=TargetResolution(width=640, height=640)).predict()

            body = self._sent_body(mock)
            self.assertEqual(body['sourceType'], 'GROUP')
            self.assertEqual(body['targetResolution'], {'width': 640, 'height': 640})

    @aioresponses()
    def test_a_source_without_a_target_resolution_sends_none(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url).predict()
            self.assertNotIn('targetResolution', self._sent_body(mock))

    @aioresponses()
    def test_upload_sends_the_target_resolution_as_a_multipart_part(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            parts: dict[str, tuple[str, bytes]] = {}

            def upload(url, **kwargs) -> CallbackResult:
                for entry in kwargs['data']:
                    part = entry[0]
                    disposition = part.headers.get('Content-Disposition', '')
                    name = disposition.split('name="', 1)[1].split('"', 1)[0]
                    parts[name] = (part.headers.get('Content-Type', ''), part._value)
                return CallbackResult(status=200, body=json.dumps(
                    {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': 0}))

            mock.post(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}'
                f'/source?mode=queue&processing=sync&version=3',
                callback=upload)

            endpoint.upload_stream(
                io.BytesIO(b'not really a png'),
                'image/png',
                target_resolution=TargetResolution(width=1280, height=720),
            ).predict()

            self.assertIn('targetResolution', parts)
            content_type, value = parts['targetResolution']
            self.assertEqual(content_type, 'application/json')
            self.assertEqual(json.loads(value), {'width': 1280, 'height': 720})


def test_pipeline_mode_set_source_carries_the_target_resolution():
    command = _set_source_command('s', WorkerSource(
        urls=['https://example.com/a.mp4'], target_resolution=TargetResolution(width=1280, height=720)))
    assert command['targetResolution'] == {'width': 1280, 'height': 720}
    assert 'targetResolution' not in _set_source_command('s', WorkerSource(urls=['https://example.com/a.mp4']))
