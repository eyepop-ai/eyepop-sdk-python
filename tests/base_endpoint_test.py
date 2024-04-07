import json
import os
import unittest
from aioresponses import aioresponses, CallbackResult


class BaseEndpointTest(unittest.IsolatedAsyncioTestCase):

    test_eyepop_url = 'http://example.test'
    test_eyepop_pop_id = 'test_pop_id'
    test_eyepop_secret_key = 'test secret key'
    test_expired_access_token = '... expired ...'
    test_access_token = '... an access token ...'
    test_worker_url = f'http://example-worker.test'
    test_pipeline_id = 'test_pipeline_id'
    test_pop_comp = "test_pop_comp"
    
    env_var = ['EYEPOP_SECRET_KEY', 'EYEPOP_POP_ID', 'EYEPOP_URL']
    for var in env_var:
        if var in os.environ:
            del os.environ[var]

    def setup_base_mock(self, mock: aioresponses, sandbox_id: str | None = None):
        mock.clear()
        def transient_config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps(
                                          {'base_url': self.test_worker_url}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps(
                                          {'base_url': self.test_worker_url, 'pipeline_id': self.test_pipeline_id}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def start(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps({'id': self.test_pipeline_id}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def stop(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=204)
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        mock.get(f'{self.test_eyepop_url}/workers/config', callback=transient_config,
                 repeat=True)
        mock.get(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True', callback=config,
                 repeat=True)
        if sandbox_id is None:
            mock.post(f'{self.test_worker_url}/pipelines', callback=start, repeat=False)
        else:
            mock.post(f'{self.test_worker_url}/pipelines?sandboxId={sandbox_id}', callback=start, repeat=False)

        mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
                   callback=stop, repeat=False)

    def assertBaseMock(self, mock: aioresponses, is_transient: bool = False, sandbox_id: str | None = None):
        mock.assert_called_with(f'{self.test_eyepop_url}/authentication/token', method='POST',
                                json={'secret_key': self.test_eyepop_secret_key})
        if is_transient:
            mock.assert_called_with(f'{self.test_eyepop_url}/workers/config',
                                    method='GET',
                                    headers={'Authorization': f'Bearer {self.test_access_token}'})
            if sandbox_id is None:
                start_url = f'{self.test_worker_url}/pipelines'
            else:
                start_url = f'{self.test_worker_url}/pipelines?sandboxId={sandbox_id}'
            mock.assert_called_with(
                start_url,
                method='POST', headers={'Authorization': f'Bearer {self.test_access_token}'}, data=None,
                json={
                'inferPipelineDef': {
                    'pipeline': 'identity'
                },
                "source": {
                    "sourceType": "NONE",
                },
                "idleTimeoutSeconds": 60,
                "logging": ["out_meta"],
                "videoOutput": "no_output"
            })
        else:
            mock.assert_called_with(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True',
                                    method='GET',
                                    headers={'Authorization': f'Bearer {self.test_access_token}'})
            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
                method='PATCH', headers={'Authorization': f'Bearer {self.test_access_token}'}, data=None,
                json={'sourceType': 'NONE'})
