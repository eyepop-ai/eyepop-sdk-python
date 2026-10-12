import json
from importlib import resources

import pytest
from aioresponses import CallbackResult, aioresponses
from pydantic import ValidationError

import tests
from eyepop import EyePopSdk
from eyepop.data.types.prediction import Prediction
from eyepop.worker.worker_types import (
    CropForward,
    ForwardOperatorType,
    FullForward,
    InferenceComponent,
    Pop,
    PopForwardOperator,
    SelectForward,
    TrackingComponent,
)
from tests.worker.base_endpoint_test import BaseEndpointTest


def face_id_pop(forward) -> Pop:
    return Pop(components=[InferenceComponent(
        ability='eyepop.person:latest',
        forward=FullForward(targets=[TrackingComponent(
            reidModel='eyepop.person.reid:latest',
            forward=forward,
        )]),
    )])


def face_targets() -> list:
    return [InferenceComponent(ability='eyepop.person.face.short-range:latest')]


def test_select_crop_round_trips_in_the_wire_shape():
    pop = face_id_pop(SelectForward(
        targets=face_targets(),
        relevancyModel='eyepop.person.face.short-range:latest',
        minTrackLengthSeconds=1,
        intervalSeconds=10,
        boxPadding=1.1,
    ))
    operator = pop.model_dump(exclude_none=True)['components'][0]['forward']['targets'][0]['forward']['operator']
    assert operator == {
        'type': 'select_crop',
        'select': {
            'mode': 'most_relevant',
            'relevancyModel': 'eyepop.person.face.short-range:latest',
            'minTrackLengthSeconds': 1,
            'intervalSeconds': 10,
        },
        'crop': {'boxPadding': 1.1},
    }
    assert Pop(**pop.model_dump()) == pop


def test_select_full_has_no_crop():
    forward = SelectForward(targets=face_targets(), full=True)
    assert forward.operator is not None
    assert forward.operator.type == ForwardOperatorType.SELECT_FULL
    assert forward.operator.crop is None


def test_select_full_rejects_crop_options():
    with pytest.raises(ValidationError, match='only valid with select_crop'):
        SelectForward(targets=face_targets(), full=True, boxPadding=1.1)


def test_select_crop_rejects_max_items():
    with pytest.raises(ValidationError, match='maxItems does not apply'):
        PopForwardOperator(type=ForwardOperatorType.SELECT_CROP, select={}, crop={'maxItems': 2})


def test_a_select_operator_requires_a_select_block():
    with pytest.raises(ValidationError, match='requires a select block'):
        PopForwardOperator(type=ForwardOperatorType.SELECT_CROP)


def test_select_belongs_only_to_a_select_operator():
    with pytest.raises(ValidationError, match='only valid with the select_crop or select_full'):
        PopForwardOperator(type=ForwardOperatorType.CROP, select={})


def test_unknown_select_mode_is_rejected():
    with pytest.raises(ValidationError):
        PopForwardOperator(type=ForwardOperatorType.SELECT_CROP, select={'mode': 'most-relevant'})


@pytest.mark.parametrize('select', [
    {'relevancyModel': 'a', 'relevancyModelUuid': 'b'},
    {'minTrackLengthSeconds': -1},
    {'intervalSeconds': 0},
])
def test_invalid_select_options_are_rejected(select):
    with pytest.raises(ValidationError):
        PopForwardOperator(type=ForwardOperatorType.SELECT_CROP, select=select)


def test_a_pop_selects_when_any_nested_forward_selects():
    assert face_id_pop(SelectForward(targets=face_targets())).selects()
    assert not face_id_pop(CropForward(targets=face_targets())).selects()
    assert not Pop(components=[]).selects()


def test_a_prediction_reads_selected():
    assert Prediction(source_width=1, source_height=1, timestamp=3, selected=True).selected
    assert Prediction(source_width=1, source_height=1).selected is None


class TestEndpointRequestsSelections(BaseEndpointTest):
    """A Pop that selects asks for prediction version 3.

    A select forward's results come only at that version, and any other Pop
    keeps asking for 2.
    """

    test_source_id = 'test_source_id'
    test_file = str(resources.files(tests) / 'test.jpg')

    def setup_pop_mock(self, mock: aioresponses, pop: Pop):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}', status=200,
                 body=json.dumps({'pop': pop.model_dump()}))

    def upload_with(self, mock: aioresponses, pop: Pop, version: int) -> list[str]:
        self.setup_pop_mock(mock, pop)
        seen: list[str] = []

        def on_post(url, **kwargs) -> CallbackResult:
            seen.append(str(url))
            return CallbackResult(status=200, body=json.dumps(
                {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': 0}))

        mock.post(
            f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}'
            f'/source?mode=queue&processing=sync&version={version}',
            callback=on_post)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            job = endpoint.upload(self.test_file)
            job.predict()
        return seen

    @aioresponses()
    def test_a_selecting_pop_asks_for_version_3(self, mock: aioresponses):
        seen = self.upload_with(mock, face_id_pop(SelectForward(targets=face_targets())), 3)
        self.assertEqual(len(seen), 1)
        self.assertIn('version=3', seen[0])

    @aioresponses()
    def test_any_other_pop_asks_for_version_2(self, mock: aioresponses):
        seen = self.upload_with(mock, face_id_pop(CropForward(targets=face_targets())), 2)
        self.assertEqual(len(seen), 1)
        self.assertIn('version=2', seen[0])
