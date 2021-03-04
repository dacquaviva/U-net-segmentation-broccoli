import os
import sys
from pathlib import Path


path_to_here = os.path.realpath(__file__)
path_to_here = path_to_here.split('.')[0]
PATH_TO_LOCAL = path_to_here.replace('scripts' + os.sep + 'boot', '')
sys.path.append(PATH_TO_LOCAL)

PATH_TO_LOCAL = Path(PATH_TO_LOCAL)
PATH_TO_PROJECT = PATH_TO_LOCAL.parent
PATH_TO_PROJECT = str(PATH_TO_PROJECT)
sys.path.append(PATH_TO_PROJECT)
