import asyncio
from viam.services.vision import Vision
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.errors import DuplicateResourceError
try:
    from src.keras_detector import KerasDetector
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .keras_detector import KerasDetector


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        Vision.API,
        KerasDetector.MODEL,
        ResourceCreatorRegistration(
            KerasDetector.new, KerasDetector.validate_config
        ),
    )
    module = Module.from_args()
    module.add_model_from_registry(Vision.API, KerasDetector.MODEL)
    await module.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except DuplicateResourceError:
        print("Duplicate resource error encountered. Restarting module...")
        asyncio.run(Module.run_from_registry())
    
