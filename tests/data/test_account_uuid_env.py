import os
import unittest
import warnings
from unittest import mock

from eyepop import EyePopSdk

BASE_ENV = {'EYEPOP_API_KEY': 'test api key', 'EYEPOP_URL': 'http://example.test'}


def account_uuid_from_env(env: dict[str, str]) -> str | None:
    with mock.patch.dict(os.environ, {**BASE_ENV, **env}, clear=True):
        endpoint = EyePopSdk.dataEndpoint(is_async=True)
    return endpoint.account_uuid


class TestAccountUuidEnv(unittest.TestCase):

    def test_reads_account_uuid(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(account_uuid_from_env({'EYEPOP_ACCOUNT_UUID': 'uuid-1'}), 'uuid-1')

    def test_account_uuid_wins_over_account_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            account_uuid = account_uuid_from_env({'EYEPOP_ACCOUNT_UUID': 'uuid-1', 'EYEPOP_ACCOUNT_ID': 'id-1'})
        self.assertEqual(account_uuid, 'uuid-1')

    def test_falls_back_to_deprecated_account_id(self):
        with self.assertWarnsRegex(DeprecationWarning, 'EYEPOP_ACCOUNT_ID is deprecated, use EYEPOP_ACCOUNT_UUID'):
            account_uuid = account_uuid_from_env({'EYEPOP_ACCOUNT_ID': 'id-1'})
        self.assertEqual(account_uuid, 'id-1')

    def test_argument_wins_over_env(self):
        with mock.patch.dict(os.environ, {**BASE_ENV, 'EYEPOP_ACCOUNT_UUID': 'uuid-1'}, clear=True):
            endpoint = EyePopSdk.dataEndpoint(account_id='arg-1', is_async=True)
        self.assertEqual(endpoint.account_uuid, 'arg-1')

    def test_unset(self):
        self.assertIsNone(account_uuid_from_env({}))
