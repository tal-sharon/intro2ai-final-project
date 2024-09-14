import abc
from collections import namedtuple
from enum import Enum

import numpy as np
import time


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5
    EXIT = 6


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def getAction(self, game_state):
        return

    def stop_running(self):
        pass
