import pandas as pd
import numpy as np
import xgboost as xgb
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
import os

#Asking the appropriate input from User.
script_path = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_path, 'stocks')
if not os.path.isdir(folder_path):
    print("Invalid folder path.")


#Function for Creating features suitable for our regression problem.
def create_features(df):
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df = df.copy()
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df

#Prediction function
def StockPredict():
    #Searches for a CSV file with the name provided by the user.
    stockname = input("Enter Stock Symbol\n")
    search_pattern = f'{stockname}.csv'
    csv_file = glob.glob(folder_path + '/' + search_pattern)
    if not csv_file:
        print("Stock Symbol not found. Please try again.\n")
        return
    #Reads the CSV and Trains our regression model based on it.
    df = pd.read_csv(csv_file[0])
    df = create_features(df)
    train = df.loc[df.index < '01-01-2019']
    test = df.loc[df.index >= '01-01-2019']
    X_train = train[['dayofyear', 'dayofweek', 'quarter', 'month', 'year']]
    y_train = train[['Close']]
    X_test = test[['dayofyear', 'dayofweek', 'quarter', 'month', 'year']]
    y_test = test[['Close']]
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
    reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
    predate = pd.DataFrame(columns=['Date'])
    #Predicts on new Data provided by the user
    predate.loc[0, 'Date'] = input("Enter the date you want to predict for. Please use the following format: yyyy-mm-dd\n")
    while True:
        try:
            # Try to parse the date string using the datetime.strptime() method
            date_obj = datetime.strptime(predate.loc[0, 'Date'], '%Y-%m-%d')
            # Check if the parsed date matches the original string
            if date_obj.strftime('%Y-%m-%d') == predate.loc[0, 'Date']:
                print("Valid date format.")
                break
            else:
                print("Invalid date format.")
                predate.loc[0, 'Date'] = input("Enter the date you want to predict for. Please use the following format: yyyy-mm-dd\n")
        except ValueError:
            print("Invalid date format.")
            predate.loc[0, 'Date'] = input("Enter the date you want to predict for. Please use the following format: yyyy-mm-dd\n")
        except Exception as e:
            print("Invalid date format.")
    predate = create_features(predate)
    print(f"The predicted closing value is: {reg.predict(predate)}")

#Reading the Stocks Meta-Data csv
metafilepath =  os.path.join(f"{folder_path}", "symbols_valid_meta.csv")
stockmeta = pd.read_csv(metafilepath)

#Program Runtime. User controls if he wants to Predict new Data or See Stock Symbols or Exit.
inp = input("Press 1 to enter Stock Symbol for predictions. Press 2 to See Stock Symbol List\n")
while True:
    if inp == "1":
        StockPredict()
        inp = input("Press 1 to enter Stock Symbol for predictions. Press 2 to See Stock Symbol List\n")
    if inp == "2":
        print(stockmeta[["Symbol", "Security Name"]])
        inp = input("Press 1 to enter Stock Symbol for predictions. Press 2 to See Stock Symbol List\n")
    else:
        break