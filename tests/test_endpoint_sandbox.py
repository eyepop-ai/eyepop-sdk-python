import os
import json
import unittest
from aiohttp import ClientResponseError
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    @aioresponses()
    def test_connect_ok(self, mock: aioresponses):

        test_sandbox_id = 'test_sandbox_id_1'
        self.setup_base_mock(mock, sandbox_id=test_sandbox_id)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        def create_sandbox(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps(test_sandbox_id))

        def delete_sandbox(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=204)

        mock.post(f'{self.test_worker_url}/sandboxes',
                    callback=create_sandbox)

        mock.delete(f'{self.test_worker_url}/sandboxes/{test_sandbox_id}',
                    callback=delete_sandbox)

        endpoint = EyePopSdk.endpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id='transient', is_sandbox=True)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock, is_transient=True, sandbox_id=test_sandbox_id)
        mock.assert_called_with(f'{self.test_worker_url}/sandboxes',
                                method='POST',
                                headers={'Authorization': f'Bearer {self.test_access_token}'})
        mock.assert_called_with(f'{self.test_worker_url}/sandboxes/{test_sandbox_id}',
                                method='DELETE',
                                headers={'Authorization': f'Bearer {self.test_access_token}'})


if __name__ == '__main__':
    unittest.main()
