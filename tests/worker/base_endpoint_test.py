import json
import os
import unittest
from asyncio import timeout

import aiohttp
from aioresponses import aioresponses, CallbackResult

from eyepop.worker.worker_types import Pop


class BaseEndpointTest(unittest.IsolatedAsyncioTestCase):

    test_eyepop_url = 'http://example.test'
    test_eyepop_pop_id = 'test_pop_id'
    test_eyepop_secret_key = 'test secret key'
    test_expired_access_token = '... expired ...'
    test_access_token = '... an access token ...'
    test_worker_url = f'http://example-worker.test'
    test_pipeline_id = 'test_pipeline_id'

    env_var = ['EYEPOP_SECRET_KEY', 'EYEPOP_POP_ID', 'EYEPOP_URL']
    for var in env_var:
        if var in os.environ:
            del os.environ[var]

    def setup_base_mock(
            self, mock: aioresponses,
            status: str = 'active_dev',
            num_endpoints: int = 1,
            provided_access_token: str | None = None,
            is_transient: bool = False,
    ):
        if provided_access_token is None:
            provided_access_token = self.test_access_token
        mock.clear()
        def transient_config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {provided_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps(
                                          {'base_url': self.test_worker_url}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {provided_access_token}':
                if status == 'active_dev':
                    return CallbackResult(status=200,
                                          body=json.dumps({
                                              'base_url': self.test_worker_url,
                                              'status': status,
                                              'pipeline_id': self.test_pipeline_id
                                          }))
                elif status == 'active_prod':
                    endpoints = []
                    for i in range(num_endpoints):
                        endpoints.append({
                            'base_url': self.test_worker_url,
                            'pipeline_id': f'{self.test_pipeline_id}-{i}'
                        })
                    return CallbackResult(status=200,
                                          body=json.dumps({
                                              'status': status,
                                              'base_url': None,
                                              'pipeline_id': None,
                                              'endpoints': endpoints
                                          }))

                else:
                    raise ValueError(f'unsupported status {status}')
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def start(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {provided_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps({'id': self.test_pipeline_id}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def stop(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {provided_access_token}':
                return CallbackResult(status=204)
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        mock.get(f'{self.test_eyepop_url}/workers/config', callback=transient_config,
                 repeat=True)
        mock.get(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True', callback=config,
                 repeat=True)
        mock.post(f'{self.test_worker_url}/pipelines', callback=start, repeat=False)

        mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
                   callback=stop, repeat=False)

        if is_transient:
            mock.delete(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}', repeat=False, status=204)

    def assertBaseMock(
            self,
            mock: aioresponses,
            is_transient: bool = False,
            status: str = 'active_dev',
            num_endpoints: int = 1,
            provided_access_token: str | None = None
    ):
        if provided_access_token is None:
            provided_access_token = self.test_access_token
            mock.assert_called_with(f'{self.test_eyepop_url}/authentication/token', method='POST',
                                    json={'secret_key': self.test_eyepop_secret_key})
        if is_transient:
            mock.assert_called_with(f'{self.test_eyepop_url}/workers/config',
                                    method='GET',
                                    headers={'Authorization': f'Bearer {provided_access_token}'})
            start_url = f'{self.test_worker_url}/pipelines'
            mock.assert_called_with(
                start_url,
                method='POST',
                headers={'Authorization': f'Bearer {provided_access_token}'},
                data=None,
                json={
                    "pop": Pop(components=[]).model_dump(),
                    "source": {
                        "sourceType": "NONE",
                    },
                    "idleTimeoutSeconds": 60,
                    "logging": ["out_meta"],
                    "videoOutput": "no_output"
                }
            )
        else:
            mock.assert_called_with(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True',
                                    method='GET',
                                    headers={'Authorization': f'Bearer {provided_access_token}'})
            if status == 'active_dev':
                mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
                method='PATCH', headers={'Authorization': f'Bearer {provided_access_token}'}, data=None,
                json={'sourceType': 'NONE'})
        if is_transient:
            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                method='DELETE', headers={'Authorization': f'Bearer {provided_access_token}'}, timeout=None)
