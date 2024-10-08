import pandas as pd 
import os 
import sys 
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from configure import config
from src import data_preprocessing, data_handling

def autoarima_inferring(data, merchant_id, predicted_month:int, path):
    forecast_value = data_preprocessing.forecasting_autorima(predicted_month, path)
    forecast_df = data_preprocessing.postpreprocessing_autoarima(data, merchant_id, predicted_month, forecast_value)
    return forecast_df

if __name__ == '__main__':
    print("Starting for AutoARIMA forecasting")
    # Loading dataset
    data = data_handling.load_dataset(config.DATA_FILE)
    # test = data[data['merchant_id'] == "0184f83cec81adfe33d4ed44385685e780c250f432dd46c71105eb7243e6a2f2"]
    arima_data = data_preprocessing.seperating_dataset(data)[1]
    # Preprocessing data
    arima_data = arima_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_outlier)
    arima_data = arima_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_missing_value)
    # Split train, test
    arima_train = arima_data[arima_data['based_month'] <= config.CUT_OFF_TRAINING_DATE]
    # Forecasting for all merchants
    autoarima_df = pd.DataFrame()
    for index, (merchant_id, merchant_data) in enumerate(arima_train.groupby('merchant_id')):
        try:
            result = autoarima_inferring(
                data = merchant_data, 
                merchant_id = merchant_id, 
                predicted_month=3,
                path = os.path.join(config.SAVE_MODEL_PATH, f'{config.MODEL_NAME}{merchant_id}.pkl') 
            )
            autoarima_df = pd.concat([autoarima_df, result], ignore_index=True)
            print(f"Finish AutoARIMA for merchant_id {merchant_id}, index = {index + 1}")
        except Exception as error:
            print(f"retailer_id {merchant_id} got {error}")
    print("Finished AutoARIMA forecasting")
    print("------------------------------")
    print("Starting for Average Revenue forecasting")
    # Loading dataset
    average_rev_data = data_preprocessing.seperating_dataset(data)[0]
    # Preprocessing data
    average_rev_data = average_rev_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_outlier)
    average_rev_data = average_rev_data.groupby('merchant_id', group_keys=False).apply(data_preprocessing.detecting_missing_value)
    # Split train, test
    average_rev_train = average_rev_data[average_rev_data['based_month'] <= config.CUT_OFF_TRAINING_DATE]
    # Forecasting for all merchants
    avg_rev_df = pd.DataFrame()
    for index, (merchant_id, merchant_data) in enumerate(average_rev_train.groupby('merchant_id')):
        try:
            result = data_preprocessing.forecasting_average_revenue(merchant_data, merchant_id, predicted_month=3)
            avg_rev_df = pd.concat([avg_rev_df, result], ignore_index=True)
            print(f"Finish Average Revenue for merchant_id {merchant_id}, index = {index + 1}")
        except Exception as error:
            print(f"retailer_id {merchant_id} got {error}")
            continue
    print("Finished Average Revenue forecasting")
    final_df = data_preprocessing.final_output(df_ma=avg_rev_df, df_ar=autoarima_df)
    print(final_df)