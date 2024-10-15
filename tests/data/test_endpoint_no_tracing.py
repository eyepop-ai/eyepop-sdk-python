import json
import unittest
from aioresponses import aioresponses

from eyepop import EyePopSdk
from .base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    @aioresponses()
    def test_no_tracing(self, mock: aioresponses):

        self.setup_base_mock(mock)

        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        endpoint = EyePopSdk.dataEndpoint(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            account_id=self.test_eyepop_account_id,
            request_tracer_max_buffer=0
        )
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)


if __name__ == '__main__':
    unittest.main()
