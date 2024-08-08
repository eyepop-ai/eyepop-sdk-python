import json
import uuid

import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, ForwardComponent, InferenceComponent, InferenceType
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
    @aioresponses()
    def test_sync_get_pop(self, mock: aioresponses):

        self.setup_base_mock(mock)
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

        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id="transient") as endpoint:
            self.assertBaseMock(mock, is_transient=True)
            cur_pop = endpoint.get_pop()
            self.assertEqual(cur_pop, self.current_pop)

    @aioresponses()
    def test_sync_set_pop(self, mock: aioresponses):
        self.setup_base_mock(mock)
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
        
        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id="transient") as endpoint:
            self.assertBaseMock(mock, is_transient=True)

            def set_pop(url, **kwargs) -> CallbackResult:
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                else:
                    self.current_pop = Pop(**json.loads(kwargs['data']))
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/pop',
                       callback=set_pop)
            
            endpoint.set_pop(self.test_new_pop)
            self.assertEqual(endpoint.get_pop(), self.test_new_pop)
            self.assertEqual(self.current_pop, self.test_new_pop)

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_get_pop(self, mock: aioresponses):

        self.setup_base_mock(mock)
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

        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id="transient", is_async=True) as endpoint:
            self.assertBaseMock(mock, is_transient=True)
            cur_pop = await endpoint.get_pop()
            self.assertIsNone(cur_pop)

    
    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_set_pop(self, mock: aioresponses):

        self.setup_base_mock(mock)
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

        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id="transient", is_async=True) as endpoint:
            self.assertBaseMock(mock, is_transient=True)

            def set_pop(url, **kwargs) -> CallbackResult:
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                elif kwargs['headers']['Content-Type'] != 'application/json':
                    return CallbackResult(status=400, reason='unsupported content type')
                else:
                    self.current_pop = Pop(**json.loads(kwargs['data']))
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/pop',
                       callback=set_pop)
            
            await endpoint.set_pop(self.test_new_pop)
            self.assertEqual(await endpoint.get_pop(), self.test_new_pop)
            self.assertEqual(self.current_pop, self.test_new_pop)