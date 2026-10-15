import unittest

from aioresponses import aioresponses

from eyepop import EyePopSdk

from .base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    @aioresponses()
    def test_no_tracing(self, mock: aioresponses):

        self.setup_base_mock(mock)

        endpoint = EyePopSdk.dataEndpoint(
            eyepop_url=self.test_eyepop_url,
            access_token=self.test_access_token,
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
