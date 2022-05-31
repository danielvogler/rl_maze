import os

class Settings:

    def __init__(self):
        self.PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.PACKAGE_ROOT)
        return