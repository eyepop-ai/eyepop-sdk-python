import json
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

    def setup_base_mock(self, mock: aioresponses):
        def config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps(
                                          {'base_url': self.test_worker_url, 'pipeline_id': self.test_pipeline_id}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def stop(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=204)
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        mock.get(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True', callback=config,
                 repeat=True)
        mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
                   callback=stop, repeat=False)

    def assertBaseMock(self, mock: aioresponses):
        mock.assert_called_with(f'{self.test_eyepop_url}/authentication/token', method='POST',
                                json={'secret_key': self.test_eyepop_secret_key})
        mock.assert_called_with(f'{self.test_eyepop_url}/pops/{self.test_eyepop_pop_id}/config?auto_start=True',
                                method='GET',
                                headers={'Authorization': f'Bearer {self.test_access_token}'})
        mock.assert_called_with(
            f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=preempt&processing=sync',
            method='PATCH', headers={'Authorization': f'Bearer {self.test_access_token}'}, data=None,
            json={'sourceType': 'NONE'})
