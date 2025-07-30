from google.protobuf.struct_pb2 import Struct
from typing import Any, List, Mapping

import pytest
import re
from src.keras_detector import KerasDetector
from viam.proto.app.robot import ComponentConfig
from viam.services.vision import CaptureAllResult, Classification, Detection, Vision
from viam.media.video import ViamImage

from tests.fake_cam import FakeCamera
from tests.fake_kerasmodel import FakeKerasModel


MODEL_PATH_ERR = "model_path must be a location (string) to a Keras model file ending in .keras"
CAMERA_ERR = "camera_name must be a non-empty string"
FAKE_CAM_NAME = "test_camera"
FAKE_MODEL_NAME = "test_model"

# Helper functions for testing
def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ComponentConfig(attributes=struct)

def getKD():
    kd = KerasDetector("test")
    kd.camera = FakeCamera(FAKE_CAM_NAME)
    print("KerasDetector created:", kd)
    return kd


# Test Validation
def test_empty_config():
    kd = getKD()
    config = make_component_config({})
    with pytest.raises(ValueError, match=re.escape(MODEL_PATH_ERR)):
        _, _= kd.validate_config(config)
    
def test_no_cam_config():
    kd = getKD()
    config = make_component_config({"model_path": "BlahBlahModel.keras"})
    with pytest.raises(ValueError, match=re.escape(CAMERA_ERR)):
        _, _= kd.validate_config(config)
    
def test_no_model_config():
    kd = getKD()
    config = make_component_config({"camera_name": FAKE_CAM_NAME})
    with pytest.raises(ValueError, match=re.escape(MODEL_PATH_ERR)):
        _, _= kd.validate_config(config)


# Test Vision Service Methods

@pytest.mark.asyncio
async def test_get_properties():
    kd = getKD()
    props = await kd.get_properties()
    assert props is not None
    assert isinstance(props, Vision.Properties)
    assert props.detections_supported
    assert not props.classifications_supported
    assert not props.object_point_clouds_supported


@pytest.mark.asyncio
async def test_detections():
    kd = getKD()
    kd.camera = FakeCamera(FAKE_CAM_NAME)
    kd.camera_name = FAKE_CAM_NAME
    kd.model = FakeKerasModel(FAKE_MODEL_NAME)
  
    img = await kd.camera.get_image()
    detections = await kd.get_detections(img)
    assert isinstance(detections, List)
    assert len(detections) > 0
    for det in detections:
        assert isinstance(det, Detection)
        assert det.x_min >= 0
        assert det.x_max >= det.x_min
        assert det.y_min >= 0
        assert det.y_max >= det.y_min


@pytest.mark.asyncio
async def test_capture_all():
    kd = getKD()
    kd.camera = FakeCamera(FAKE_CAM_NAME)
    kd.camera_name = FAKE_CAM_NAME
    kd.model = FakeKerasModel(FAKE_MODEL_NAME)

    capture_result = await kd.capture_all_from_camera(
        camera_name=FAKE_CAM_NAME,
        return_image=True,
        return_detections=True
    )
    assert isinstance(capture_result, CaptureAllResult)
    assert capture_result.image is not None
    assert isinstance(capture_result.image, ViamImage)
    assert isinstance(capture_result.detections, List)
    assert len(capture_result.detections) > 0
    assert isinstance(capture_result.detections[0], Detection)