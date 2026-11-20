"""A session's recordings by mediaId (eyepop-instance spec 025): the `recording` event, the instance's reads, warnings.

A source that records gets `{"event": {"type": "recording", "mediaId": "v2...."}}` as the first line of its
result stream, and its recording is read from the session's instance at `GET /media/{mediaId}/range|frame|clip`,
at the instance's root rather than under the pipeline.
"""
import json
import logging
import re
import time

import pytest
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.exceptions import MediaReadException
from eyepop.settings import settings
from eyepop.worker.worker_types import Pop
from tests.worker.base_endpoint_test import BaseEndpointTest

MEDIA_ID = 'v2.AAAAAAAAAAAAAAAAAAAAAA.c2Vzc2lvbg'
GAP = {'message': 'the source went 12.0 s without a keyframe',
       'info': {'code': 'media_cache_gop_too_long', 'level': 'warn', 'gap_ms': 12000, 'threshold_ms': 10000}}
PERSON = {'classLabel': 'person', 'x': 10.4, 'y': 20.0, 'width': 30.2, 'height': 40.0}


class TestMediaReadsOnASession(BaseEndpointTest):
    test_source_id = 'test_source_id'
    test_url = 'rtsp://camera.invalid/stream'

    def _setup_worker(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}),
            repeat=True)
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                 body=json.dumps({'pop': Pop(components=[]).model_dump()}))

    def _mock_source(self, mock: aioresponses, lines: list[dict]):
        def load_from(url, **kwargs) -> CallbackResult:
            return CallbackResult(status=200, body='\n'.join(json.dumps(line) for line in lines) + '\n')

        mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                   callback=load_from)

    def _media_url(self, read: str) -> re.Pattern:
        return re.compile(re.escape(f'{self.test_worker_url}/media/{MEDIA_ID}/{read}') + r'(\?.*)?$')

    def _reads(self, mock: aioresponses, read: str) -> list:
        """Each read of `read`, as (url with its query, the request's keyword arguments)."""
        return [(url, call.kwargs)
                for (method, url), calls in mock.requests.items() if method == 'GET'
                and url.path == f'/media/{MEDIA_ID}/{read}'
                for call in calls]

    def _prediction(self) -> dict:
        return {'timestamp': 33_366_666, 'source_width': 640, 'source_height': 480, 'objects': [PERSON]}

    @aioresponses()
    async def test_the_recording_event_and_a_warning_are_kept_on_the_job(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            prediction = self._prediction()
            self._mock_source(mock, [
                {'event': {'type': 'recording', 'mediaId': MEDIA_ID}},
                {'event': 'warning', 'source_id': self.test_source_id, **GAP},
                prediction,
            ])
            with self.assertLogs('eyepop', logging.WARNING) as logs:
                job = await endpoint.load_from(self.test_url, media_cache_seconds=600)
                self.assertEqual(await job.predict(), prediction)
                self.assertIsNone(await job.predict())
            self.assertEqual(job.media_id, MEDIA_ID)
            self.assertEqual(job.warnings, [GAP])
            self.assertTrue(any('12.0 s without a keyframe' in line for line in logs.output))

    @aioresponses()
    async def test_a_source_that_does_not_record_has_no_media_id(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            prediction = self._prediction()
            self._mock_source(mock, [prediction])
            job = await endpoint.load_from(self.test_url)
            self.assertEqual(await job.predict(), prediction)
            self.assertIsNone(job.media_id)
            self.assertEqual(job.warnings, [])
            with self.assertRaisesRegex(ValueError, 'no media_id'):
                await job.media_frame(prediction, PERSON)

    @aioresponses()
    async def test_reads_go_to_the_instance_by_media_id(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            prediction = self._prediction()
            self._mock_source(mock, [{'event': {'type': 'recording', 'mediaId': MEDIA_ID}}, prediction])
            job = await endpoint.load_from(self.test_url, media_cache_seconds=600)
            await job.predict()

            mock.get(self._media_url('range'), payload={'start_ns': 1_000, 'end_ns': 9_000})
            mock.get(self._media_url('frame'), body=b'\xff\xd8 a jpeg', content_type='image/jpeg', repeat=True)
            self.assertEqual(await endpoint.media_range(MEDIA_ID), (1_000, 9_000))
            self.assertEqual(await endpoint.media_frame(MEDIA_ID, 5_000, bbox=(1, 2, 3, 4), max_width=160,
                                                        png=True, wait_ms=0), b'\xff\xd8 a jpeg')
            self.assertEqual(await job.media_frame(prediction, PERSON, max_height=80), b'\xff\xd8 a jpeg')

        (_, range_kwargs), = self._reads(mock, 'range')
        self.assertEqual(range_kwargs['headers']['Authorization'], f'Bearer {self.test_access_token}')
        (_, frame), (_, helper) = self._reads(mock, 'frame')
        self.assertEqual(frame['params'], {'timestamp': 5_000, 'bbox': '1,2,3,4', 'maxWidth': 160, 'wait': 0})
        self.assertEqual(frame['headers']['Accept'], 'image/png')
        # the prediction's timestamp, the smallest box of whole pixels that holds the object's, JPEG
        self.assertEqual(helper['params'], {'timestamp': 33_366_666, 'bbox': '10,20,31,40', 'maxHeight': 80})
        self.assertEqual(helper['headers']['Accept'], 'image/jpeg')

    @aioresponses()
    async def test_a_clip_reports_its_bounds_and_no_units(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            mock.get(self._media_url('clip'), body=b'an mp4', content_type='video/mp4',
                     headers={'X-Start-Ns': '1999', 'X-End-Ns': '4000'})
            clip = await endpoint.media_clip(MEDIA_ID, 2_000, 4_000, mode='keyframe')
            self.assertEqual((clip.start_ns, clip.end_ns, clip.milli_compute_units), (1_999, 4_000, None))
            self.assertEqual(await clip.read(), b'an mp4')
        (_, kwargs), = self._reads(mock, 'clip')
        self.assertEqual(kwargs['params'], {'start': 2_000, 'end': 4_000, 'mode': 'keyframe'})

    @aioresponses()
    async def test_a_425_is_asked_again_after_retry_after(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            mock.get(self._media_url('frame'), status=425, headers={'Retry-After': '0'},
                     payload={'error': 'not yet', 'reason': 'media_not_yet_available', 'message': 'wait'})
            mock.get(self._media_url('frame'), body=b'a jpeg', content_type='image/jpeg')
            started = time.monotonic()
            self.assertEqual(await endpoint.media_frame(MEDIA_ID, 5_000), b'a jpeg')
            self.assertGreaterEqual(time.monotonic() - started, 0.4, 'Retry-After 0 is waited at least 0.5 s')
        self.assertEqual(len(self._reads(mock, 'frame')), 2)

    @aioresponses()
    async def test_a_425_past_the_deadline_and_a_404_raise(self, mock: aioresponses):
        self._setup_worker(mock)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            mock.get(self._media_url('clip'), status=425, headers={'Retry-After': '2'},
                     payload={'error': 'not yet', 'reason': 'media_not_yet_available', 'message': 'wait'})
            previous = settings.max_retry_time_secs
            settings.max_retry_time_secs = 1.0
            try:
                with self.assertRaises(MediaReadException) as raised:
                    await endpoint.media_clip(MEDIA_ID, 2_000, 4_000)
            finally:
                settings.max_retry_time_secs = previous
            self.assertEqual((raised.exception.status, raised.exception.reason, raised.exception.retry_after),
                             (425, 'media_not_yet_available', 2.0))

            mock.get(self._media_url('range'), status=404, payload={
                'error': 'recording not found', 'reason': 'stream_not_found',
                'message': 'no recording of this session has this mediaId'})
            with self.assertRaises(MediaReadException) as raised:
                await endpoint.media_range(MEDIA_ID)
            self.assertEqual((raised.exception.status, raised.exception.error, raised.exception.reason),
                             (404, 'recording not found', 'stream_not_found'))
            self.assertIn('no recording of this session', str(raised.exception))

    @aioresponses()
    async def test_a_load_balanced_pop_reads_from_its_worker(self, mock: aioresponses):
        self.setup_base_mock(mock, status='active_prod')
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}),
            repeat=True)
        async with EyePopSdk.async_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                          pop_id=self.test_eyepop_pop_id) as endpoint:
            mock.get(self._media_url('range'), payload={'start_ns': 1, 'end_ns': 2})
            self.assertEqual(await endpoint.media_range(MEDIA_ID), (1, 2))

    @aioresponses()
    def test_the_sync_endpoint_reads_too(self, mock: aioresponses):
        self._setup_worker(mock)
        with EyePopSdk.sync_worker(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                   pop_id=self.test_eyepop_pop_id) as endpoint:
            prediction = self._prediction()
            self._mock_source(mock, [{'event': {'type': 'recording', 'mediaId': MEDIA_ID}}, prediction])
            job = endpoint.load_from(self.test_url, media_cache_seconds=600)
            self.assertEqual(job.predict(), prediction)
            self.assertEqual(job.media_id, MEDIA_ID)

            mock.get(self._media_url('range'), payload={'start_ns': 1_000, 'end_ns': 9_000})
            mock.get(self._media_url('frame'), body=b'a jpeg', content_type='image/jpeg', repeat=True)
            mock.get(self._media_url('clip'), body=b'an mp4', content_type='video/mp4',
                     headers={'X-Start-Ns': '2000', 'X-End-Ns': '4000'}, repeat=True)
            self.assertEqual(endpoint.media_range(MEDIA_ID), (1_000, 9_000))
            self.assertEqual(endpoint.media_frame(MEDIA_ID, 5_000), b'a jpeg')
            self.assertEqual(job.media_frame(prediction, PERSON), b'a jpeg')
            clip = endpoint.media_clip(MEDIA_ID, 2_000, 4_000)
            self.assertEqual((clip.start_ns, clip.end_ns), (2_000, 4_000))
            self.assertEqual(clip.read(), b'an mp4')
            with endpoint.media_clip(MEDIA_ID, 2_000, 4_000) as clip:
                self.assertEqual(b''.join(clip), b'an mp4')


@pytest.mark.parametrize(('bbox', 'expected'), [
    ((1, 2, 3, 4), '1,2,3,4'),
    ((10.4, 20.0, 30.2, 40.0), '10,20,31,40'),
    ((-0.5, 0.5, 1.0, 1.0), '-1,0,2,2'),
])
def test_a_box_is_sent_in_whole_pixels_that_hold_it(bbox, expected):
    from eyepop.worker.media import bbox_param
    assert bbox_param(bbox) == expected

