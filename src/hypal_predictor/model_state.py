from enum import Enum, auto


class ModelState(Enum):
    INITIALIZING = auto()
    GATHERING = auto()
    TRAINING = auto()
    READY = auto()
