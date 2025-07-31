from typing import Any, Coroutine, Final, List, Optional, Tuple, Dict
from viam.components.camera import Camera
from viam.gen.component.camera.v1.camera_pb2 import GetPropertiesResponse
from viam.media.video import NamedImage, ViamImage, CameraMimeType
from viam.media.utils import pil
from viam.proto.common import ResponseMetadata
from PIL import Image


# FakeCamera is a mock implementation of the Camera component for testing purposes.
# It returns a 640 x 480 image which is notably different from the expected Keras input size (320x180)
class FakeCamera(Camera):

    def __init__(self, name: str):
        super().__init__(name=name)
        self.img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        
    async def get_image(self, mime_type: str = "") -> Coroutine[Any, Any, ViamImage]:
        return pil.pil_to_viam_image(self.img, CameraMimeType.JPEG)

    async def get_images(self) -> Coroutine[Any, Any, Tuple[List[NamedImage] | ResponseMetadata]]:
        raise NotImplementedError

    async def get_properties(self) -> Coroutine[Any, Any, GetPropertiesResponse]:
        raise NotImplementedError

    async def get_point_cloud(self) -> Coroutine[Any, Any, Tuple[bytes | str]]:
        raise NotImplementedError
    
