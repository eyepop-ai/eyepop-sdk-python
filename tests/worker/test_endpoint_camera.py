import io
import json
import time

import aiohttp
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.camera import Camera, CameraExtrinsics, CameraIntrinsics, Quaternion, Vector3d
from eyepop.worker.worker_types import DEFAULT_PREDICTION_VERSION, Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointCamera(BaseEndpointTest):
    """A per-source camera reaches the worker on the routes that start a source."""

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

    @aioresponses()
    def test_load_from_sends_a_full_calibration(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            camera = Camera(
                intrinsics=CameraIntrinsics(fx=0.9, fy=1.6, cx=0.5, cy=0.5),
                extrinsics=CameraExtrinsics(rotation=Quaternion(w=1.0), translation=Vector3d(z=3.0)),
            )
            endpoint.load_from(self.test_url, camera=camera).predict()

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
                    'camera': {
                        'intrinsics': {'fx': 0.9, 'fy': 1.6, 'cx': 0.5, 'cy': 0.5},
                        'extrinsics': {
                            'rotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'translation': {'x': 0.0, 'y': 0.0, 'z': 3.0},
                        },
                    },
                }),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    def test_load_from_sends_the_field_of_view_shorthand(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url, camera=Camera(hfovDegrees=72.0)).predict()

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
                    'camera': {'hfovDegrees': 72.0},
                }),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    def test_a_source_without_a_camera_is_unchanged(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self._mock_source(mock)
            endpoint.load_from(self.test_url).predict()

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=json.dumps({'sourceType': 'URL', 'url': self.test_url,
                                 'version': DEFAULT_PREDICTION_VERSION}),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    def test_upload_sends_the_camera_as_a_multipart_part(self, mock: aioresponses):
        """The upload routes carry the calibration as an application/json part.

        Named parts beside `params`, `roi` and `fps`, which is the shape the
        instance accepts; a filename is set because that is what puts the part
        among the files rather than the form values.
        """
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
                f'/source?mode=queue&processing=sync&version=2',
                callback=upload)

            endpoint.upload_stream(
                io.BytesIO(b'not really a png'),
                'image/png',
                camera=Camera(hfovDegrees=72.0),
            ).predict()

            self.assertIn('camera', parts)
            content_type, value = parts['camera']
            self.assertEqual(content_type, 'application/json')
            self.assertEqual(json.loads(value), {'hfovDegrees': 72.0})
