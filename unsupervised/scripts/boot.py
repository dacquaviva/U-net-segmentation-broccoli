import os
import sys

path_to_here = os.path.realpath(__file__)
path_to_here = path_to_here.split('.')[0]
PATH_TO_PROJECT = path_to_here.replace('scripts' + os.sep + 'boot', '')
sys.path.append(PATH_TO_PROJECT)

