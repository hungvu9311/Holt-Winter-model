from tqdm import tqdm
import os 
import sys 
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from configure import config
from src import data_preprocessing, data_handling

def autoarima_training_params(merchant_data, merchant_id, season_length):
    # Training model
    model_fit = data_preprocessing.training_autoarima(merchant_data, season_length)
    return data_handling.save_params(model_fit, merchant_id)

if __name__ == '__main__':
    print("Starting for training models")
    # Loading dataset
    data = data_handling.load_dataset(config.DATA_FILE)
    arima_data = data_preprocessing.test_seperating_dataset(data)[1]
    # Preprocessing data
    arima_data = arima_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_outlier)
    arima_data = arima_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_missing_value)
    # Split train, test
    arima_train = arima_data[arima_data['based_month'] <= config.CUT_OFF_TRAINING_DATE]

    # Finding best param for each merchant id
    for index, (merchant_id, merchant_data) in tqdm(enumerate(arima_train.groupby('merchant_id'))):
        autoarima_training_params(merchant_data, merchant_id, config.SEASON_LENGTH_PARAM)
        print(f'Finish finding the best param for merchant_id {merchant_id}, index = {index + 1}')
    print("Finished training the models")    