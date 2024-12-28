from datetime import datetime
import json
import os
import unittest
from aioresponses import aioresponses, CallbackResult

from eyepop.data.data_types import DatasetResponse


class BaseEndpointTest(unittest.IsolatedAsyncioTestCase):
    test_eyepop_url = 'http://example.test'
    test_eyepop_account_id = 'test_account_id'
    test_eyepop_secret_key = 'test secret key'
    test_expired_access_token = '... expired ...'
    test_access_token = '... an access token ...'
    test_data_url = f'http://example-data.test'
    test_dataset_id = 'test_dataset_id'

    test_dataset = DatasetResponse(
        uuid=test_dataset_id,
        name="test",
        description="",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        tags=[],
        account_uuid=test_eyepop_account_id,
        auto_annotates=[],
        versions=[]
    )

    env_var = ['EYEPOP_SECRET_KEY', 'EYEPOP_ACCOUNT_ID', 'EYEPOP_URL']
    for var in env_var:
        if var in os.environ:
            del os.environ[var]

    def setup_base_mock(self, mock: aioresponses):
        mock.clear()

        def config(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=json.dumps({"base_url": self.test_data_url}))
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def create_dataset(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=200,
                                      body=self.test_dataset.model_dump_json())
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        def delete_dataset(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                return CallbackResult(status=204)
            else:
                return CallbackResult(status=401, reason='test auth token expired')

        mock.get(f'{self.test_eyepop_url}/data/config?account_uuid={self.test_eyepop_account_id}', callback=config, repeat=True)
        mock.post(f'{self.test_data_url}/datasets?account_uuid={self.test_eyepop_account_id}', callback=create_dataset, repeat=False)
        mock.delete(f'{self.test_data_url}/datasets/{self.test_dataset_id}', callback=delete_dataset, repeat=False)

    def assertBaseMock(self, mock: aioresponses):
        mock.assert_called_with(f'{self.test_eyepop_url}/authentication/token', method='POST',
                                json={'secret_key': self.test_eyepop_secret_key})
        # mock.assert_called_with(
        #     f'{self.test_data_url}datasets?account_uuid={self.test_eyepop_account_id}',
        #     method='POST', headers={'Authorization': f'Bearer {self.test_access_token}'}, data=None,
        #     json=self.test_dataset.dict())
