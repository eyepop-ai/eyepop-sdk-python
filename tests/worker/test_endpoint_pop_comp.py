import json
import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointPopComp(BaseEndpointTest):
    test_pop_comp = "test_pop_comp"
    test_new_pop_comp = "test_new_pop_comp"

    @aioresponses()
    def test_sync_get_pop_comp(self, mock: aioresponses):

        self.setup_base_mock(mock)
        # authentication
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)

        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock)
            cur_pop_comp = endpoint.get_pop_comp()
            self.assertEqual(cur_pop_comp, self.test_pop_comp)

    @aioresponses()
    def test_sync_set_pop_comp(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        
        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock)

            pop_comp = self.test_pop_comp

            def set_pop_comp(url, **kwargs) -> CallbackResult:
                nonlocal pop_comp
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                else:
                    pop_comp = json.loads(kwargs['data'])['pipeline']
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/inferencePipeline',
                       callback=set_pop_comp)
            
            endpoint.set_pop_comp(self.test_new_pop_comp)
            self.assertEqual(endpoint.get_pop_comp(), self.test_new_pop_comp)
            self.assertEqual(pop_comp, self.test_new_pop_comp)

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_get_pop_comp(self, mock: aioresponses):

        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        
        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id, is_async=True) as endpoint:
            self.assertBaseMock(mock)
            cur_pop_comp = await endpoint.get_pop_comp()
            self.assertEqual(cur_pop_comp, self.test_pop_comp)

    
    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_set_pop_comp(self, mock: aioresponses):

        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        
        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id, is_async=True) as endpoint:
            self.assertBaseMock(mock)

            pop_comp = self.test_pop_comp

            def set_pop_comp(url, **kwargs) -> CallbackResult:
                nonlocal pop_comp
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                elif kwargs['headers']['Content-Type'] != 'application/json':
                    return CallbackResult(status=400, reason='unsupported content type')
                else:
                    pop_comp = json.loads(kwargs['data'])['pipeline']
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/inferencePipeline',
                       callback=set_pop_comp)
            
            await endpoint.set_pop_comp(self.test_new_pop_comp)
            self.assertEqual(await endpoint.get_pop_comp(), self.test_new_pop_comp)
            self.assertEqual(pop_comp, self.test_new_pop_comp)