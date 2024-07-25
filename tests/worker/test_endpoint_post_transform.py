import json
import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointPostTransform(BaseEndpointTest):
    test_post_transform = "test_post_transform"
    test_new_post_transform = "test_new_post_transform"

    @aioresponses()
    def test_sync_get_post_transform(self, mock: aioresponses):

        self.setup_base_mock(mock)
        # authentication
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_post_transform(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'postTransform': self.test_post_transform}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_post_transform)

        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock)
            cur_post_transform = endpoint.get_post_transform()
            self.assertEqual(cur_post_transform, self.test_post_transform)

    @aioresponses()
    def test_sync_set_post_transform(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_post_transform(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'postTransform': self.test_post_transform}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_post_transform)
        
        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock)

            transform = self.test_post_transform

            def set_post_transform(url, **kwargs) -> CallbackResult:
                nonlocal transform
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                else:
                    transform = json.loads(kwargs['data'])['transform']
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/postTransform',
                       callback=set_post_transform)
            
            endpoint.set_post_transform(self.test_new_post_transform)
            self.assertEqual(endpoint.get_post_transform(), self.test_new_post_transform)
            self.assertEqual(transform, self.test_new_post_transform)


    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_get_post_transform(self, mock: aioresponses):

        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_post_transform(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'postTransform': self.test_post_transform}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_post_transform)
        
        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id, is_async=True) as endpoint:
            self.assertBaseMock(mock)
            cur_post_transform = await endpoint.get_post_transform()
            self.assertEqual(cur_post_transform, self.test_post_transform)

    
    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_set_post_transform(self, mock: aioresponses):

        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_post_transform(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'postTransform': self.test_post_transform}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_post_transform)

        transform = self.test_post_transform

        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id, is_async=True) as endpoint:
            self.assertBaseMock(mock)

            def set_post_transform(url, **kwargs) -> CallbackResult:
                nonlocal transform
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                elif kwargs['headers']['Content-Type'] != 'application/json':
                    return CallbackResult(status=400, reason='unsupported content type')
                else:
                    transform = json.loads(kwargs['data'])['transform']
                    return CallbackResult(status=204)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/postTransform',
                       callback=set_post_transform)
            
            await endpoint.set_post_transform(self.test_new_post_transform)
            self.assertEqual(await endpoint.get_post_transform(), self.test_new_post_transform)
            self.assertEqual(transform, self.test_new_post_transform)