import os
import pandas as pd
import pickle
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.configure import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_parquet(filepath, engine="fastparquet")
    return _data

def save_params(model_fit, merchant_id):
    save_path = os.path.join(config.SAVE_MODEL_PATH, f'{config.MODEL_NAME}{merchant_id}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model_fit, f)

def load_params(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model