import pandas as pd 
from statsforecast.models import AutoARIMA
import os 
import sys 
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.configure import config
from prediction_model.src import data_preprocessing, data_handling

#--------------------------------DATA PREPROCESSING-----------------------------------
def seperating_dataset(df):
    # counting df point of each merchant
    df_point = df.groupby('merchant_id')['based_month'].count()
    # seperating list of merchant
    less_than_13m = df_point[(df_point >= 4) & (df_point <= 13)].index
    more_than_13m = df_point[df_point > 13].index

    df_ma = df[df['merchant_id'].isin(less_than_13m)]
    df_hw = df[df['merchant_id'].isin(more_than_13m)]
    return df_ma, df_hw

def detecting_outlier(df):
    Q1 = df['net_revenue'].quantile(0.25)
    Q3 = df['net_revenue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df.loc[df['net_revenue'] < lower_bound, 'net_revenue'] = lower_bound
    df.loc[df['net_revenue'] > upper_bound, 'net_revenue'] = upper_bound

    return df

def detecting_missing_value(df): 
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        if i == 0:
            continue
        mean_previous_month = df.loc[:i-1,'net_revenue'].mean()
        if (df.loc[i,'net_revenue'] == 0) or (df.loc[i,'net_revenue'] < 0):
            df.loc[i,'net_revenue'] = mean_previous_month
    return df

def indexing_date(df):
    df = df.set_index('based_month')
    return df

#--------------------------------MODEL FUNCTION-----------------------------------
def training_autoarima(df, season_length: int):
    model = AutoARIMA(season_length=season_length)
    model_fit = model.fit(df['net_revenue'].values)
    return model_fit

def forecasting_autorima(predicted_month: int, path):
    model = data_handling.load_params(path)
    forecast_value = model.predict(predicted_month)['mean']
    return forecast_value

def postpreprocessing_autoarima(data, merchant_id, predicted_month: int, forecast_value):
    # Create a new date for the predicted month
    last_month = data['based_month'].max()
    # Generate future dates for the next 3 months
    future_dates = pd.date_range(last_month + pd.DateOffset(months=1), periods=predicted_month, freq='MS')
    forecast_df = pd.DataFrame({
        'merchant_id': merchant_id,
        'based_month': future_dates,
        'forecast_revenue': list(forecast_value),
        'model': 'autoarima'
    })
    # Adjust high-value and negative output 
    for i in range(0, len(forecast_df)):
        mean_previous_months = forecast_df.loc[:i-1, 'forecast_revenue'].mean()
        std_previous_months = forecast_df.loc[:i-1, 'forecast_revenue'].std()
        threshold = mean_previous_months + 3 * std_previous_months #rule of very high : >= 3 standard deviations
        if (forecast_df.loc[i, 'forecast_revenue'] > threshold):  #threshold for 'very high' or 'negative' value
            forecast_df.loc[i, 'forecast_revenue'] = mean_previous_months 
        elif forecast_df.loc[i, 'forecast_revenue'] < 0:
            forecast_df.loc[i, 'forecast_revenue'] = 0
    return forecast_df

def forecasting_average_revenue(data, merchant_id, predicted_month: int):
    # Calculate the average of all previous revenues for the merchant
    mean_revenue = data['net_revenue'].tail(predicted_month).mean()
    # Create a new date for the predicted month
    last_month = data['based_month'].max()
    # Generate future dates for the next 3 months
    future_dates = pd.date_range(last_month + pd.DateOffset(months=1), periods=predicted_month, freq='MS')

    # Create a new DataFrame for the future months with the average revenue
    future_df = pd.DataFrame({
        'merchant_id': merchant_id, 
        'based_month': future_dates,
        'net_revenue': [mean_revenue] * predicted_month,
        'model': 'average_revenue_l3m'
    })                   
    future_df = future_df.rename(columns={'net_revenue' : 'forecast_revenue'})
    return future_df



#----------------------------MERGE OUTPUTS---------------------------
def final_output(df_ma, df_ar):
    final_df = pd.concat([df_ma, df_ar], ignore_index=True, sort=False)
    return final_df

