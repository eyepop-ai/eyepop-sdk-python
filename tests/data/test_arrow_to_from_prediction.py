import json
import unittest
from importlib import resources

from parameterized import parameterized

from eyepop.data.arrow.eyepop.predictions import (
    eyepop_predicted_classes_from_table,
    eyepop_predicted_objects_from_table,
    table_from_eyepop_predicted_classes,
    table_from_eyepop_predicted_objects,
)
from eyepop.data.data_types import Prediction

from . import files


class TestArrowToFromPrediction(unittest.TestCase):
    @parameterized.expand([
        ("prediction_0_bbox.json", 1),
        ("prediction_1_bbox.json", 2),
        ("prediction_2_bbox.json", 3),
    ])
    def test_prediction_objects_from_file(self, file_name, n):
        test_json = resources.files(files) / file_name
        with test_json.open("r") as f:
            source_prediction = Prediction(**json.load(f))
        table = table_from_eyepop_predicted_objects(
            source_prediction.objects,
            source_prediction.source_width,
            source_prediction.source_height
        )
        assert table is not None
        assert table.num_rows == len(source_prediction.objects)
        target_predicted_objects = eyepop_predicted_objects_from_table(
            table,
            source_prediction.source_width,
            source_prediction.source_height
        )
        assert target_predicted_objects is not None
        if source_prediction.objects != target_predicted_objects:
            raise AssertionError()

    @parameterized.expand([
        ("prediction_3_classes.json", 4),
    ])
    def test_prediction_classes_from_file(self, file_name, n):
        test_json = resources.files(files) / file_name
        with test_json.open("r") as f:
            source_prediction = Prediction(**json.load(f))
        table = table_from_eyepop_predicted_classes(
            source_prediction.classes
        )
        assert table is not None
        assert table.num_rows == len(source_prediction.classes)
        target_predicted_classes = eyepop_predicted_classes_from_table(
            table,
        )
        assert target_predicted_classes is not None
        if source_prediction.classes != target_predicted_classes:
            raise AssertionError()
