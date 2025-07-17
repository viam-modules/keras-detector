import asyncio
from viam.module.module import Module
try:
    from models.keras_detector import KerasDetector
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.keras_detector import KerasDetector


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
