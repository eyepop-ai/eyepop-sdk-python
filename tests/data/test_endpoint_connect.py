import unittest

from aiohttp import ClientResponseError
from aioresponses import aioresponses

from eyepop import EyePopSdk
from tests.data.base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    def test_missing_secret(self):
        with self.assertRaises(KeyError):
            EyePopSdk.dataEndpoint()

    @aioresponses()
    def test_connect_ok(self, mock: aioresponses):

        self.setup_base_mock(mock)

        endpoint = EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, access_token=self.test_access_token,
                                          account_id=self.test_eyepop_account_id)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)

    @aioresponses()
    def test_connect_unauthorized(self, mock: aioresponses):
        self.setup_base_mock(mock)

        endpoint = EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, access_token=self.test_expired_access_token,
                                          account_id=self.test_eyepop_account_id)
        with self.assertRaises(ClientResponseError) as context:
            try:
                endpoint.connect()
            finally:
                endpoint.disconnect()
        self.assertEqual(context.exception.status, 401)


if __name__ == '__main__':
    unittest.main()
