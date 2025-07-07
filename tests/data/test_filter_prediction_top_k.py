import json
from importlib import resources

import pytest
from eyepop.data.data_types import Prediction

from eyepop.data.data_top_k import filter_prediction_top_k
from . import files

class TestFilterPrediction:
    @pytest.mark.parametrize("file_name, has_classes, has_objects, n, k", [
        ("prediction_10_classes.json", True, False, 10, 5),
        ("prediction_10_objects.json", False, True, 10, 5),
        ("prediction_10_classes_10_objects.json", True, True, 10, 5),
        ("prediction_10_classes_10_objects.json", True, True, 10, 15),
    ])
    def test_filter_prediction_from_file(self, file_name, has_classes: bool, has_objects: bool, n: int, k: int):
        test_json = resources.files(files) / file_name
        with test_json.open("r") as f:
            source_prediction = Prediction(**json.load(f))
        assert has_classes == (True if source_prediction.classes else False)
        assert has_objects == (True if source_prediction.objects else False)
        if has_classes:
            assert n == len(source_prediction.classes)
        if has_objects:
            assert n == len(source_prediction.objects)

        filtered_prediction = filter_prediction_top_k(source_prediction, k)
        assert has_classes == (True if filtered_prediction.classes else False)
        assert has_objects == (True if filtered_prediction.objects else False)
        if has_classes:
            assert min(n, k) == len(filtered_prediction.classes)
            for clazz in filtered_prediction.classes:
                assert clazz in source_prediction.classes
            for clazz in source_prediction.classes:
                if clazz not in filtered_prediction.classes:
                    for filtered_clazz in filtered_prediction.classes:
                        assert clazz.confidence is None or filtered_clazz.confidence >= clazz.confidence

        if has_objects:
            assert min(n, k) == len(filtered_prediction.objects)
            for obj in filtered_prediction.objects:
                assert obj in source_prediction.objects
            for obj in source_prediction.objects:
                if obj not in filtered_prediction.objects:
                    for filtered_obj in filtered_prediction.objects:
                        assert obj.confidence is None or filtered_obj.confidence >= obj.confidence
