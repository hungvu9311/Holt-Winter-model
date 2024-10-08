from pathlib import Path
import os 

PACKAGE_ROOT = Path(os.getcwd())
# Data path
DATAPATH = os.path.join(PACKAGE_ROOT, "data\\raw")
DATA_FILE = "all_data_revenue.parquet"

# AutoARIMA variables
MODEL_NAME = 'autoarima_'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_param')
CUT_OFF_TRAINING_DATE = '2024-06-01'
SEASON_LENGTH_PARAM = 12


VARIABLE_SEASONAL_PARAM = ['add','mul']
VARIABLE_TREND_PARAM = ['add','mul']
VARIABLE_SEASONAL_PERIOD = list(range(2,13))

print(DATAPATH)
