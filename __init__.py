import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from node import Eden_LoRa_trainer

NODE_CLASS_MAPPINGS = {
    "Eden_LoRa_trainer": Eden_LoRa_trainer,
}

