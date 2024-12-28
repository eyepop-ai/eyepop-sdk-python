import json
import unittest
import uuid

from aiohttp import ClientResponseError
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    def test_missing_secret(self):
        with self.assertRaises(Exception):
            EyePopSdk.workerEndpoint()

    @aioresponses()
    def test_connect_ok(self, mock: aioresponses):

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
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)

    @aioresponses()
    def test_connect_with_token_ok(self, mock: aioresponses):

        provided_access_token = uuid.uuid4().hex

        self.setup_base_mock(mock, provided_access_token=provided_access_token)

        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {provided_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, access_token=provided_access_token,
                                            pop_id=self.test_eyepop_pop_id)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock, provided_access_token=provided_access_token)

    @aioresponses()
    def test_connect_transient_ok(self, mock: aioresponses):

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
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id='transient')
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock, True)

    @aioresponses()
    def test_connect_unauthorized(self, mock: aioresponses):
        self.setup_base_mock(mock)

        mock.post(f'{self.test_eyepop_url}/authentication/token', status=401, body="test unauthorized")
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id)
        with self.assertRaises(ClientResponseError) as context:
            try:
                endpoint.connect()
            finally:
                endpoint.disconnect()
        self.assertEqual(context.exception.status, 401)

    @aioresponses()
    def test_connect_reauth_expired(self, mock: aioresponses):

        self.setup_base_mock(mock)

        auth_calls = 0

        def first_auth(url, **kwargs) -> CallbackResult:
            nonlocal auth_calls
            auth_calls += 1
            return CallbackResult(status=200, body=json.dumps(
                {'expires_in': 1, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        def second_auth(url, **kwargs) -> CallbackResult:
            nonlocal auth_calls
            auth_calls += 1
            return CallbackResult(status=200, body=json.dumps(
                {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        mock.post(f'{self.test_eyepop_url}/authentication/token', callback=first_auth)
        mock.post(f'{self.test_eyepop_url}/authentication/token', callback=second_auth)
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)

        self.assertEqual(auth_calls, 2)

    @aioresponses()
    def test_connect_reauth_401(self, mock: aioresponses):
        self.setup_base_mock(mock)

        auth_calls = 0

        def auth(url, **kwargs) -> CallbackResult:
            nonlocal auth_calls
            if auth_calls == 0:
                result = CallbackResult(status=200, body=json.dumps(
                    {'expires_in': 1000, 'token_type': 'Bearer', 'access_token': self.test_expired_access_token}))
            else:
                result = CallbackResult(status=200, body=json.dumps(
                    {'expires_in': 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
            auth_calls += 1
            return result

        mock.post(f'{self.test_eyepop_url}/authentication/token', callback=auth, repeat=True)
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        endpoint = EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id)
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)
        self.assertEqual(auth_calls, 2)


if __name__ == '__main__':
    unittest.main()
