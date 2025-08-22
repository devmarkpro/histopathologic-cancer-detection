import torch
from device import Device
import logging


logger = logging.getLogger(__name__)


def main():
    logger.debug("Hello from histopathologic-cancer-detection!")
    torch.manual_seed(42)
    Device.set_device()




if __name__ == "__main__":
    main()
