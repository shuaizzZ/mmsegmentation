
import sys
from mmcv.runner import Hook

class CheckRunstateHook(Hook):
    def __init__(self, runstate):
        self.runstate = runstate

    def before_iter(self, runner):
        if self.runstate[0] == 0:
            sys.exit(0)