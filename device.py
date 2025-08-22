import torch
import logging

logger = logging.getLogger(__name__)


class Device:
    """
    Device class to manage the device settings for PyTorch operations.
    This class provides methods to set and get the device based on the availability of CUDA or MPS.
    It ensures that the device is set to either 'cuda', 'mps', or 'cpu' based on the system's capabilities.
    """

    _device = torch.device("cpu")

    @classmethod
    def set_device(cls) -> torch.device:
        """
        Set the device for PyTorch based on availability of CUDA or MPS.
        Returns:
            torch.device: The device to be used for PyTorch operations.
        """

        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
        if torch.backends.mps.is_available():
            device_name = "mps"
        logger.debug(f"Using device: {device_name}")

        cls._device = torch.device(device_name)
        return cls._device

    @classmethod
    def get_device(cls) -> torch.device:
        """
        @classmethod
        def get_device(cls) -> torch.device:
            Get the current device being used by PyTorch.
            Returns:
                torch.device: The current device.
        """
        return cls._device


if __name__ == "__main__":
    Device.set_device()
    logger.debug(f"Current device: {Device.get_device()}")
