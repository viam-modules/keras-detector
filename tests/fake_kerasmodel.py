from typing import Any, Coroutine, Final, List, Optional, Tuple, Dict
from viam.components.camera import Camera
from viam.gen.component.camera.v1.camera_pb2 import GetPropertiesResponse
from viam.media.video import NamedImage, ViamImage, CameraMimeType
from viam.media.utils import pil
from viam.proto.common import ResponseMetadata
from PIL import Image



class FakeKerasModel():

    def __init__(self, name: str):
        self.name = name
        

    def predict(self, np_array, verbose: int = 0):
        # Simulate a prediction output
        return [[10, 20, 30, 40]] 

    