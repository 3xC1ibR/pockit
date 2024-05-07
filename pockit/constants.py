import os

repo_root = os.path.dirname(os.path.dirname(__file__))


data = os.path.join(repo_root, 'data')
models = os.path.join(repo_root, 'models')

deeplab = os.path.join(models, 'deeplab.h5')

from .utils import file_exists
file_exists(deeplab)
