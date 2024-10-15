import json
import unittest
from datetime import datetime

from aiohttp import ClientResponseError
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from eyepop.data.data_types import DatasetCreate, DatasetResponse, DatasetUpdate, AutoAnnotate
from tests.data.base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):

    def test_missing_secret(self):
        with self.assertRaises(Exception):
            EyePopSdk.dataEndpoint()

    @aioresponses()
    def test_crud_dataset(self, mock: aioresponses):

        self.setup_base_mock(mock)

        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        mock.post(f'{self.test_data_url}/datasets?account_uuid={self.test_eyepop_account_id}', status=200,
                  body=DatasetResponse(
                    uuid=self.test_dataset_id,
                    name="",
                    description="",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    tags=[],
                    account_uuid=self.test_eyepop_account_id,
                    auto_annotates=[],
                    auto_annotate_params=None,
                    versions=[],
        ).model_dump_json())
        mock.patch(f'{self.test_data_url}/datasets/{self.test_dataset_id}?start_auto_annotate=True', status=200, body=DatasetResponse(
            uuid=self.test_dataset_id,
            name="",
            description="",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            account_uuid=self.test_eyepop_account_id,
            auto_annotates=[],
            auto_annotate_params=None,
            versions=[],
        ).model_dump_json())
        mock.delete(f'{self.test_data_url}/datasets/{self.test_dataset_id}', status=204)

        with EyePopSdk.dataEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                    account_id=self.test_eyepop_account_id) as endpoint:
            endpoint.connect()
            dataset = endpoint.create_dataset(DatasetCreate(
                name="test"
            ))
            dataset = endpoint.update_dataset(dataset.uuid, DatasetUpdate(
                name="test updated",
                auto_anotates=["ep_coco"]
            ))
            endpoint.delete_dataset(dataset.uuid)

        self.assertBaseMock(mock)
        mock.assert_called_with(
            f'{self.test_data_url}/datasets?account_uuid={self.test_eyepop_account_id}',
            method='POST',
            timeout=None,
            headers={
                'Authorization': f'Bearer {self.test_access_token}',
                'Content-Type': 'application/json',
            },
            data=DatasetCreate(
                name="test"
            ).model_dump_json()
        )
        mock.assert_called_with(
            f'{self.test_data_url}/datasets/{self.test_dataset_id}?start_auto_annotate=True',
            method='PATCH',
            timeout=None,
            headers={
                'Authorization': f'Bearer {self.test_access_token}',
                'Content-Type': 'application/json',
            },
            data=DatasetUpdate(
                name="test updated",
                auto_anotates=["ep_coco"]
            ).model_dump_json(exclude_none=True, exclude_unset=True)
        )
        mock.assert_called_with(
            f'{self.test_data_url}/datasets/{self.test_dataset_id}',
            method='DELETE',
            timeout=None,
            data=None,
            headers={'Authorization': f'Bearer {self.test_access_token}'}
        )


if __name__ == '__main__':
    unittest.main()
