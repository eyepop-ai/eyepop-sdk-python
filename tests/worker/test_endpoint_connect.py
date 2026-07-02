import asyncio
import json
import unittest
import uuid
from types import MethodType

from aiohttp import ClientResponseError
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointConnect(BaseEndpointTest):
    def test_missing_secret(self):
        with self.assertRaises(KeyError):
            EyePopSdk.sync_worker()

    def test_session_name_sets_compute_context(self):
        endpoint = EyePopSdk.async_worker(
            eyepop_url="https://compute.eyepop.ai",
            api_key="test-api-key",
            pop_id="transient",
            session_name="sessions-smoke-123",
        )

        self.assertIsNotNone(endpoint.compute_ctx)
        self.assertEqual(endpoint.compute_ctx.session_name, "sessions-smoke-123")

    async def test_ensure_pipeline_started_serializes_concurrent_creation(self):
        endpoint = object.__new__(WorkerEndpoint)
        endpoint.worker_config = {}
        endpoint._pipeline_create_lock = asyncio.Lock()
        create_calls = 0

        async def create_pipeline(self):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)
            self.worker_config["pipeline_id"] = "created-pipeline"
            return {"id": "created-pipeline"}

        endpoint._create_pipeline = MethodType(create_pipeline, endpoint)

        await asyncio.gather(
            endpoint._ensure_pipeline_started(),
            endpoint._ensure_pipeline_started(),
        )

        self.assertEqual(create_calls, 1)
        self.assertEqual(endpoint.worker_config["pipeline_id"], "created-pipeline")

    @aioresponses()
    def test_connect_ok(self, mock: aioresponses):
        self.setup_base_mock(mock)

        mock.post(
            f"{self.test_eyepop_url}/authentication/token",
            status=200,
            body=json.dumps(
                {
                    "expires_in": 1000 * 1000,
                    "token_type": "Bearer",
                    "access_token": self.test_access_token,
                }
            ),
        )

        # automatic call to get pop comp and store
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {self.test_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            pop_id=self.test_eyepop_pop_id,
        )
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
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {provided_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            access_token=provided_access_token,
            pop_id=self.test_eyepop_pop_id,
        )
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock, provided_access_token=provided_access_token)

    @aioresponses()
    def test_connect_transient_ok(self, mock: aioresponses):
        self.setup_base_mock(mock, is_transient=True)

        mock.post(
            f"{self.test_eyepop_url}/authentication/token",
            status=200,
            body=json.dumps(
                {
                    "expires_in": 1000 * 1000,
                    "token_type": "Bearer",
                    "access_token": self.test_access_token,
                }
            ),
        )

        # automatic call to get pop comp and store
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {self.test_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            pop_id="transient",
        )
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock, is_transient=True, expect_pipeline_started=False)

    @aioresponses()
    async def test_set_pop_on_compute_transient_rechecks_session_scheduling(
        self, mock: aioresponses
    ):
        new_pop = Pop(components=[])
        captured_session_body = {}
        captured_pipeline_body = {}
        session_response = {
            "session_uuid": "session-456",
            "session_endpoint": self.test_worker_url,
            "access_token": self.test_access_token,
            "pipelines": [{"pipeline_id": self.test_pipeline_id}],
            "session_status": "running",
        }
        session_response_without_pipeline = {
            **session_response,
            "pipelines": [],
        }

        def create_or_update_session(url, **kwargs) -> CallbackResult:
            captured_session_body.update(kwargs["json"])
            return CallbackResult(status=200, body=json.dumps(session_response_without_pipeline))

        def create_pipeline(url, **kwargs) -> CallbackResult:
            captured_pipeline_body.update(kwargs["json"])
            return CallbackResult(status=200, body=json.dumps({"id": self.test_pipeline_id}))

        mock.get(
            f"{self.test_eyepop_url}/v1/sessions", status=200, body=json.dumps([session_response])
        )
        mock.post(
            f"{self.test_eyepop_url}/v1/sessions?wait=true", callback=create_or_update_session
        )
        mock.get(
            f"{self.test_worker_url}/health",
            status=200,
            body=json.dumps({"message": "ok"}),
            repeat=True,
        )
        mock.delete(
            f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", status=204, repeat=True
        )
        mock.post(f"{self.test_worker_url}/pipelines", callback=create_pipeline)
        mock.get(
            f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}",
            status=200,
            body=json.dumps({"pop": new_pop.model_dump(), "status": "IDLE"}),
            repeat=True,
        )

        endpoint = EyePopSdk.async_worker(
            eyepop_url=self.test_eyepop_url,
            api_key="test-api-key",
            pop_id="transient",
            stop_jobs=False,
        )
        try:
            await endpoint.connect()
            response = await endpoint.set_pop(new_pop)

            self.assertEqual(response["id"], self.test_pipeline_id)
            self.assertEqual(captured_session_body["pop"], new_pop.model_dump())
            self.assertEqual(captured_pipeline_body["pop"], new_pop.model_dump())
        finally:
            await endpoint.disconnect()

        mock.assert_called_with(
            f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}",
            method="DELETE",
            headers={"Authorization": f"Bearer {self.test_access_token}"},
            timeout=None,
        )
        mock.assert_called_with(
            f"{self.test_eyepop_url}/v1/sessions?wait=true",
            method="POST",
            headers={
                "Authorization": "Bearer test-api-key",
                "Accept": "application/json",
            },
            json={"pop": new_pop.model_dump()},
        )

    @aioresponses()
    def test_connect_unauthorized(self, mock: aioresponses):
        self.setup_base_mock(mock)

        mock.post(
            f"{self.test_eyepop_url}/authentication/token", status=401, body="test unauthorized"
        )

        # automatic call to get pop comp and store
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {self.test_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            pop_id=self.test_eyepop_pop_id,
        )
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
            return CallbackResult(
                status=200,
                body=json.dumps(
                    {
                        "expires_in": 1,
                        "token_type": "Bearer",
                        "access_token": self.test_access_token,
                    }
                ),
            )

        def second_auth(url, **kwargs) -> CallbackResult:
            nonlocal auth_calls
            auth_calls += 1
            return CallbackResult(
                status=200,
                body=json.dumps(
                    {
                        "expires_in": 1000 * 1000,
                        "token_type": "Bearer",
                        "access_token": self.test_access_token,
                    }
                ),
            )

        mock.post(f"{self.test_eyepop_url}/authentication/token", callback=first_auth)
        mock.post(f"{self.test_eyepop_url}/authentication/token", callback=second_auth)

        # automatic call to get pop comp and store
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {self.test_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            pop_id=self.test_eyepop_pop_id,
        )
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
                result = CallbackResult(
                    status=200,
                    body=json.dumps(
                        {
                            "expires_in": 1000,
                            "token_type": "Bearer",
                            "access_token": self.test_expired_access_token,
                        }
                    ),
                )
            else:
                result = CallbackResult(
                    status=200,
                    body=json.dumps(
                        {
                            "expires_in": 1000,
                            "token_type": "Bearer",
                            "access_token": self.test_access_token,
                        }
                    ),
                )
            auth_calls += 1
            return result

        mock.post(f"{self.test_eyepop_url}/authentication/token", callback=auth, repeat=True)

        # automatic call to get pop comp and store
        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs["headers"]["Authorization"] != f"Bearer {self.test_access_token}":
                return CallbackResult(status=401, reason="test auth token expired")
            else:
                return CallbackResult(
                    status=200, body=json.dumps({"pop": Pop(components=[]).model_dump()})
                )

        mock.get(f"{self.test_worker_url}/pipelines/{self.test_pipeline_id}", callback=get_pop)
        endpoint = EyePopSdk.sync_worker(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            pop_id=self.test_eyepop_pop_id,
        )
        try:
            endpoint.connect()
        finally:
            endpoint.disconnect()

        self.assertBaseMock(mock)
        self.assertEqual(auth_calls, 2)


if __name__ == "__main__":
    unittest.main()
