import unittest

from eyepop.data.data_types import Prediction, PredictedClass


class TestPredictionMetadata(unittest.TestCase):
    """Test new metadata fields: source_id and seconds."""

    def test_prediction_with_new_fields(self):
        """Test Prediction model with source_id and seconds."""
        pred = Prediction(
            source_width=1920.0,
            source_height=1080.0,
            timestamp=1000000000,  # 1 second in nanoseconds
            source_id="test-asset-uuid",
            seconds=1.0,
            classes=[PredictedClass(classLabel="Exterior")]
        )

        assert pred.source_id == "test-asset-uuid"
        assert pred.seconds == 1.0
        assert pred.timestamp == 1000000000

    def test_prediction_without_new_fields(self):
        """Test backward compatibility - new fields are optional."""
        pred = Prediction(
            source_width=1920.0,
            source_height=1080.0
        )

        assert pred.source_id is None
        assert pred.seconds is None

    def test_prediction_serialization(self):
        """Test JSON serialization with new fields."""
        pred = Prediction(
            source_width=612.0,
            source_height=408.0,
            timestamp=0,
            seconds=0.0,
            source_id="bb97f3ec-f6f5-11f0-9250-5ecb5ad6a98e",
            classes=[PredictedClass(classLabel="Exterior")]
        )

        json_data = pred.model_dump(exclude_none=True)
        assert json_data["source_id"] == "bb97f3ec-f6f5-11f0-9250-5ecb5ad6a98e"
        assert json_data["seconds"] == 0.0
        assert json_data["timestamp"] == 0

    def test_prediction_with_url_as_source_id(self):
        """Test source_id with URL."""
        pred = Prediction(
            source_width=1024.0,
            source_height=768.0,
            source_id="https://example.com/image.jpg",
            seconds=0.0
        )

        assert pred.source_id == "https://example.com/image.jpg"

    def test_prediction_video_frame_seconds(self):
        """Test seconds field for video frames."""
        # Frame 0 at time 0
        pred0 = Prediction(
            source_width=1920.0,
            source_height=1080.0,
            timestamp=0,
            seconds=0.0,
            offset=0
        )
        assert pred0.seconds == 0.0

        # Frame 1 at 30fps = 33.33ms = 0.0333 seconds
        pred1 = Prediction(
            source_width=1920.0,
            source_height=1080.0,
            timestamp=33333333,  # ~33ms in nanoseconds
            seconds=0.033,
            offset=1
        )
        assert abs(pred1.seconds - 0.033) < 0.001  # Allow small float precision error


if __name__ == "__main__":
    unittest.main()
