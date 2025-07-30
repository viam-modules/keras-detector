import os
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf
from typing_extensions import Self
from viam.logging import getLogger
from viam.media.video import ViamImage, CameraMimeType
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.components.camera import Camera
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes



class KerasDetector(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "vision"), "keras-detector"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)


    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        model_path_err = "model_path must be a location (string) to a Keras model file ending in .keras"
        model_path = config.attributes.fields["model_path"].string_value

        if model_path is None or model_path == "":
            raise ValueError(model_path_err)
        _, ext = os.path.splitext(model_path)
        if ext != ".keras":
            raise ValueError(model_path_err)

        camera_name = config.attributes.fields["camera_name"].string_value
        if camera_name is None or camera_name == "":
            raise ValueError("camera_name must be a non-empty string")

        return [camera_name], []


    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both required and optional)
        """

        self.logger = getLogger("keras_detector")

        self.model_path = config.attributes.fields["model_path"].string_value
        self.model = keras.models.load_model(self.model_path)

        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]

        return super().reconfigure(config, dependencies)


    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        
        out = CaptureAllResult()

        if camera_name not in [self.camera_name, ""]:
            raise ValueError(f"Camera {camera_name} is not the configured camera {self.camera_name}")
        img = await self.camera.get_image(CameraMimeType.JPEG)

        if return_image:
            out.image = img
        if return_detections:
            out.detections = await self.get_detections(img, extra=extra, timeout=timeout)
        # No classifications
        # No object point clouds
        return out
        
 
    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        
        if camera_name != self.camera_name:
            raise ValueError(f"Camera {camera_name} is not the configured camera {self.camera_name}")
        
        viam_img = await self.camera.get_image(CameraMimeType.JPEG)
        
        return self.get_detections(viam_img, extra=extra, timeout=timeout)
        

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        
        pil_img = viam_to_pil_image(image)
        img = self.prep_image(pil_img)

        # Send the image thru the model
        out = self.model.predict(img, verbose=0)
        out_dets = []

        for o in out:
            if len(o) < 4:
                self.logger.warning("this doesn't seem like a valid detection, skipping")
                continue
            det = Detection(
                x_min=min(round(o[0]), round(o[2])),  
                y_min=min(round(o[1]), round(o[3])),  
                x_max=max(round(o[0]), round(o[2])),  
                y_max=max(round(o[1]), round(o[3])),  
                class_name="object",
                confidence=0.5
            )
            out_dets.append(det)
        
        return out_dets


    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError()


    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")
        raise NotImplementedError()


    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()


    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
        out = Vision.Properties(
            detections_supported=True,
            classifications_supported=False,
            object_point_clouds_supported=False
        )
        return out


    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()
    

    # prep_image returns the image as a numpy with the batch dimension in front.
    def prep_image(self, input_image, target_size=(320, 180, 3)):
        image = keras.utils.img_to_array(input_image)
        image = np.resize(image, target_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  
        return image
