import json
import uuid

import pytest
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.worker_types import ForwardComponent, InferenceComponent, InferenceType, Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointPop(BaseEndpointTest):
    test_pop = Pop(
        components=[ForwardComponent()]
    )
    test_new_pop = Pop(
        components=[InferenceComponent(
            inferenceTypes=[InferenceType.OBJECT_DETECTION],
            modelUuid=uuid.uuid4().hex
        )]
    )

    current_pop: Pop | None = None

    def assert_pipeline_started_with_pop(self, mock: aioresponses, pop: Pop):
        mock.assert_called_with(
            f'{self.test_worker_url}/pipelines',
            method='POST',
            headers={'Authorization': f'Bearer {self.test_access_token}'},
            data=None,
            json={
                "pop": pop.model_dump(),
                "source": {
                    "sourceType": "NONE",
                },
                "idleTimeoutSeconds": 300,
                "logging": ["out_meta"],
                "videoOutput": "no_output"
            }
        )

    @aioresponses()
    def test_sync_get_pop(self, mock: aioresponses):

        self.setup_base_mock(mock, is_transient=True)
        # authentication
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pipeline(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'pop': self.current_pop}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pipeline)

        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id="transient",
        ) as endpoint:
            cur_pop = endpoint.get_pop()
        self.assertEqual(cur_pop, Pop(components=[]))
        self.assertBaseMock(mock, is_transient=True, expect_pipeline_started=False)

    @aioresponses()
    def test_sync_set_pop(self, mock: aioresponses):
        self.setup_base_mock(mock, is_transient=True)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pipeline(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'pop': self.current_pop}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pipeline)
        
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id="transient",
        ) as endpoint:
            endpoint.set_pop(self.test_new_pop)
            self.assertEqual(endpoint.get_pop(), self.test_new_pop)
            self.assert_pipeline_started_with_pop(mock, self.test_new_pop)


    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_get_pop(self, mock: aioresponses):

        self.setup_base_mock(mock, is_transient=True)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pipeline(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'pop': self.current_pop}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pipeline)

        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id="transient",
        ) as endpoint:
            cur_pop = await endpoint.get_pop()
        self.assertEqual(cur_pop, Pop(components=[]))
        self.assertBaseMock(mock, is_transient=True, expect_pipeline_started=False)

    
    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_set_pop(self, mock: aioresponses):

        self.setup_base_mock(mock, is_transient=True)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pipeline(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'pop': self.current_pop}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pipeline)

        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id="transient",
        ) as endpoint:
            await endpoint.set_pop(self.test_new_pop)
            self.assertEqual(await endpoint.get_pop(), self.test_new_pop)
            self.assert_pipeline_started_with_pop(mock, self.test_new_pop)
