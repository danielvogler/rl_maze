'''
Daniel Vogler

rlmaze.py

'''


from .settings import Settings

class RLMaze:

    def __init__(self):
        self.PROJECT_ROOT = Settings().PROJECT_ROOT
        self.PACKAGE_ROOT = Settings().PACKAGE_ROOT
        return