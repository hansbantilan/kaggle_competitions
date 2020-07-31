from kaggle.api.kaggle_api_extended import KaggleApi #https://github.com/Kaggle/kaggle-api
import os

# Use Kaggle API to download train.csv, test.csv
api = KaggleApi()
api.authenticate()
api.competition_download_files('nlp-getting-started')

# Unzip files
os.system('unzip *.zip');
