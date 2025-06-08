import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = '\\data'

TRAIN_CSV = os.path.join(BASE_DIR + DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(BASE_DIR + DATA_FOLDER, 'val.csv')


