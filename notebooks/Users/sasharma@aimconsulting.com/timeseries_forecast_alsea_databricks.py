# Databricks notebook source
#ingestion
from awscli.customizations.s3.utils import split_s3_bucket_key
import boto3 
import pyarrow.parquet as pq
import io

# formatting & analysis 
import pandas as pd
import numpy as np
import glob
from datetime import datetime as dt
from datetime import timedelta 
from datetime import date
from time import time

# viz
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

# prediction
import xgboost
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

# processing
# import dask.dataframe as dd

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os

# TODO update horrendous naming convention
# TODO multicollinearity likely causing inflated results; test removal of more variables
# TODO reduce packages listed to only those utilized
# TODO make offsets primary features
# TODO add MultiOutputRegressor 


# COMMAND ----------

def alter_datetimes_type(transaction_df): 
    '''function to convert type obj date and time to type datetime while 
        perserving original frame format'''
   
    t_df = transaction_df

    if (isinstance(t_df['DATE_D'].iloc[0], date) == False): # if date col is not of type datetime
        # parse date int YYYYMMDD to datetime; reorder to MMDDYYYY; format to string    
        t_df['DATE_D'] = t_df['DATE_D'].apply(lambda x: dt.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d')) 
        # combine date string and time string  
        t_df['date_time'] = pd.to_datetime(t_df['DATE_D'] + ' ' + t_df['TIME_D'])
        # t_df['date_time'] = t_df['date_time'].to_datetime()
        # parse YYYYMMDD string to date  
        t_df['DATE_D'] = t_df['DATE_D'].apply(lambda x: dt.strptime(x, '%Y-%m-%d').date())
        

    return t_df


# COMMAND ----------

def create_date_features(df): 

    df['year'] = df.date_time.apply(lambda date: date.year)
    df['month'] = df.date_time.apply(lambda date: date.month)
    df['day'] = df.date_time.apply(lambda date: date.day)
    df['dayofweek'] = df.date_time.apply(lambda date: date.dayofweek)
    df['hour'] = df.date_time.apply(lambda time: time.hour)
    df['minute'] = df.date_time.apply(lambda time: time.minute)


    def weekend(row):
        if row.dayofweek == 5 or row.dayofweek == 6:
            return 1
        else:
            return 0

    weekend_days = df.date_time.apply(weekend)
    df['weekend'] = weekend_days

    return df


# COMMAND ----------

def format_agg_forecast_input(df):
    '''function to group and format the columns to be ingested; 
    granularity of forecast should be specified *prior* to 
    calling format_product_forecast_input'''

    groups = ['year', 'month', 'day', 'hour']
    
    #ensure correct dtype... in the future this should be handled via data classes
    df['AMOUNT'] = df['AMOUNT'].astype(float)
    df['PARTY_SIZE'] = df['PARTY_SIZE'].astype(float)
    
    aggregations = {
                    'AMOUNT': ['sum', 'max', 'min', 'mean', 'std','count'],
                    'PARTY_SIZE': ['max', 'min', 'mean', 'std']
                    }

    input_df = df.groupby(groups).agg(aggregations).reset_index()

    columns_names = ['year','month','day','hour','max_spend',
                    'total_spend','min_spend','avg_spend','std_spend','transaction_count',
                    'max_grp_size','min_grp_size','avg_grp_size','std_grp_size']

    input_df.columns = columns_names

    # recreate datetime index for altering sample sizes
    input_df['timestamp'] = pd.to_datetime(input_df[['year', 'month', 'day', 'hour']])
    input_df = input_df.set_index('timestamp')
    
    return input_df

# COMMAND ----------

def format_store_forecast_input(df):
    '''function to group and format the columns to be ingested; 
    granularity of forecast should be specified *prior* to 
    calling format_product_forecast_input'''

    groups = ['ST_NUM','year', 'month', 'day', 'hour']
    
    aggregations = {
                    'AMOUNT': ['sum', 'max', 'min', 'mean', 'std','count'],
                    'PARTY_SIZE': ['max', 'min', 'mean', 'std']
                    }

    input_df = df.groupby(groups).agg(aggregations).reset_index()

    columns_names = ['store_id','year','month','day','hour','max_spend',
                    'total_spend','min_spend','avg_spend','std_spend','transaction_count',
                    'max_grp_size','min_grp_size','avg_grp_size','std_grp_size']

    input_df.columns = columns_names

    # recreate datetime index for altering sample sizes
    input_df['timestamp'] = pd.to_datetime(input_df[['year', 'month', 'day', 'hour']])
    input_df = input_df.set_index('timestamp')
    
    return input_df

# COMMAND ----------

def format_product_forecast_input(df):
    '''function to group and format the columns to be ingested; 
    granularity of forecast should be specified *prior* to 
    calling format_product_forecast_input'''

    groups = ['PRODUCT_WID','year', 'month', 'day', 'hour']
    
    aggregations = {
                    'AMOUNT': ['sum', 'max', 'min', 'mean', 'std','count'],
                    'PARTY_SIZE': ['max', 'min', 'mean', 'std']
                    }

    input_df = df.groupby(groups).agg(aggregations).reset_index()

    columns_names = ['product_id','year','month','day','hour','max_spend',
                    'total_spend','min_spend','avg_spend','std_spend','transaction_count',
                    'max_grp_size','min_grp_size','avg_grp_size','std_grp_size']

    input_df.columns = columns_names

    # recreate datetime index for altering sample sizes
    input_df['timestamp'] = pd.to_datetime(input_df[['year', 'month', 'day', 'hour']])
    input_df = input_df.set_index('timestamp')
    
    return input_df

# COMMAND ----------

def resample_index_hour(input_df):
    '''assumes datetime index in format YYYYMMDD r'''

    r = input_df.resample('1H') # set resample level; creates DatetimeIndexResampler object

    aggregations = { # TODO clean resample group method; 
                    'year': np.mean, #intended to preserve column 
                    'month': np.mean, # intended to preserve column 
                    'day': np.mean, # intended to preserve column 
                    'hour': np.mean, # intended to preserve column 
                    'total_spend': np.sum,  
                    'max_spend': np.mean, # intended to preserve column 
                    'min_spend': np.mean, # intended to preserve column 
                    'avg_spend': np.mean, # intended to preserve column 
                    'std_spend': np.mean, # intended to preserve column 
                    'transaction_count': np.sum, # intended to preserve column 
                    'max_grp_size': np.mean, # intended to preserve column 
                    'min_grp_size': np.mean, # intended to preserve column 
                    'avg_grp_size': np.mean, # intended to preserve column 
                    'std_grp_size': np.mean # intended to preserve column 
                    }

    resampled_df = r.agg(aggregations)
    resampled_df = resampled_df.interpolate(method='linear')

    return resampled_df
  
def encode_categorical_hour(df):
  '''must be paired with respective resample function '''

  columns_to_category = ['year', 'month', 'day','hour']
  # data[columns_to_category] = data[columns_to_category].astype('category') 
  # ^^^ only necessary if categorical values present in columns_to_category
  data = pd.get_dummies(df, columns=columns_to_category) # one hot encoding 

  return data

# COMMAND ----------

def resample_index_day(input_df):
    '''assumes datetime index in format YYYYMMDD r'''

    r = input_df.resample('1D') # set resample level; creates DatetimeIndexResampler object; 
    # fill missing dates w/median between date range sampled

    aggregations = { # TODO clean resample group method; 
                    'year': np.mean, #intended to preserve column 
                    'month': np.mean, # intended to preserve column 
                    'day': np.mean, # intended to preserve column 
                    'total_spend': np.sum, 
                    'max_spend': np.mean, # intended to preserve column 
                    'min_spend': np.mean, # intended to preserve column 
                    'avg_spend': np.mean, # intended to preserve column 
                    'std_spend': np.mean, # intended to preserve column 
                    'transaction_count': np.sum, # intended to preserve column 
                    'max_grp_size': np.mean, # intended to preserve column 
                    'min_grp_size': np.mean, # intended to preserve column 
                    'avg_grp_size': np.mean, # intended to preserve column 
                    'std_grp_size': np.mean # intended to preserve column 
                    }

    resampled_df = r.agg(aggregations)

    return resampled_df

def encode_categorical_day(df):
    '''must be paired with respective resample function '''
    
    columns_to_category = ['year', 'month', 'day']
    # data[columns_to_category] = data[columns_to_category].astype('category') 
    # ^^^ only necessary if categorical values present in columns_to_category
    data = pd.get_dummies(df, columns=columns_to_category) # one hot encoding 

    return data

# COMMAND ----------

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# COMMAND ----------

def timeseries_train_test_split(X, y, test_size):
    '''perform train-test split with respect to time series structure'''  
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


# COMMAND ----------

def plotModelResults(model, X_train, X_test, plot_intervals=False, plot_anomalies=False):
    '''plots modelled vs fact values, prediction intervals and anomalies'''
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)


# COMMAND ----------

def plotCoefficients(model):
    ''' plots sorted coefficient values of the model'''
     
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')

# COMMAND ----------

def create_aggregate_timeseries_output(input_df):

    prediction_dates = []
    revenue_prediction = []
    # df_sub = pd.DataFrame([])

    day_input = resample_index_day(input_df)

    target_col = 'total_spend' # NOTE change your target here

    data = pd.DataFrame(day_input[target_col].copy())
    data.columns = ["y"]
    data.y = data.y.where(data.y!=0) # converts target vars equalling 0 to NaN for future interpolation
    data.y = data.y.interpolate(method='time') # method works on daily and higher resolution data to interpolate given length of interval
    # data = data[data.y.between(*data.y.quantile([0.05, 0.95]).tolist())] # TODO this removes some dates... 

    lag_start = 6
    lag_stop = 20

    for i in range(lag_start, lag_stop):
        data["lag_{}".format(i)] = data.y.shift(i)

    data['year'] = day_input['year']
    data['month'] = day_input['month']
    data['day'] = day_input['day']
    data["weekday"] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5,6])*1
    data['transaction_count'] = day_input['transaction_count']
    data['avg_grp_size'] = day_input['avg_grp_size']

    tscv = TimeSeriesSplit(n_splits=5) # TODO dynamic split selection

    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
    # used to transform into same scale given features vary in range
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb = XGBRegressor()
    xgb.fit(X_train_scaled, y_train)

    predictions = xgb.predict(X_test_scaled)

    prediction_dates = y_test.index
    revenue_prediction = predictions

    output = pd.DataFrame({
                    "run_date": date.today(),
                    "prediction_dates": prediction_dates,
                    "est_total_revenue": revenue_prediction
                    })


    return output

# COMMAND ----------

def create_product_timeseries_output(dim_list, input_df):

    prediction_dates = []
    revenue_prediction = []
    # df_sub = pd.DataFrame([])

    dim_dict = {}
    output_dict = {}
    
    # convert transaction df into dict based on dim 
    # iterate through dict 

    # group by dimension 

    for dim in dim_list:
        dim_dict[dim] = input_df[(input_df['product_id'] == dim)]
        day_input = resample_index_day(dim_dict[dim])

        target_col = 'total_spend' # NOTE change your target here

        data = pd.DataFrame(day_input[target_col].copy())
        data.columns = ["y"]
        data.y = data.y.where(data.y!=0) # converts target vars equalling 0 to NaN for future interpolation
        data.y = data.y.interpolate(method='time') # method works on daily and higher resolution data to interpolate given length of interval
        data = data[data.y.between(*data.y.quantile([0.05, 0.95]).tolist())] # TODO this removes some dates... may need to alter

        lag_start = 6
        lag_stop = 20

        for i in range(lag_start, lag_stop):
            data["lag_{}".format(i)] = data.y.shift(i)

        data['year'] = day_input['year']
        data['month'] = day_input['month']
        data['day'] = day_input['day']
        data["weekday"] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5,6])*1
        data['transaction_count'] = day_input['transaction_count']
        data['avg_grp_size'] = day_input['avg_grp_size']

        tscv = TimeSeriesSplit(n_splits=5) # TODO dynamic split selection

        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
        # used to transform into same scale given features vary in range
        scaler = StandardScaler()

        # necessary to handle sparsity; TODO move fucntion up stream to improve performance
        if (X_train.shape[0] <= 2) or (X_test.shape[0] <= 2):
            prediction_dates = y_test.index
            revenue_prediction = [None] * X_test.shape[0]
            
            output = pd.DataFrame({
                        "run_date": date.today(),
                        "prediction_dates": prediction_dates,
                        "est_total_revenue": revenue_prediction
                        })

            output_dict[dim] = output
            next
        
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            xgb = XGBRegressor()
            xgb.fit(X_train_scaled, y_train)

            predictions = xgb.predict(X_test_scaled)

            prediction_dates = y_test.index
            revenue_prediction = predictions

            output = pd.DataFrame({
                            "run_date": date.today(),
                            "prediction_dates": prediction_dates,
                            "est_total_revenue": revenue_prediction
                            })

            output_dict[dim] = output

    return output_dict

# COMMAND ----------

def create_store_timeseries_output(dim_list, input_df):

    prediction_dates = []
    revenue_prediction = []
    # df_sub = pd.DataFrame([])

    dim_dict = {}
    output_dict = {}
    
    # convert transaction df into dict based on dim 
    # iterate through dict 

    # group by dimension 

    for dim in dim_list:
        dim_dict[dim] = input_df[(input_df['store_id'] == dim)] 
        day_input = resample_index_day(dim_dict[dim])

        target_col = 'total_spend' # NOTE change your target here

        data = pd.DataFrame(day_input[target_col].copy())
        data.columns = ["y"]
        data.y = data.y.where(data.y!=0) # converts target vars equalling 0 to NaN for future interpolation
        data.y = data.y.interpolate(method='time') # method works on daily and higher resolution data to interpolate given length of interval
        data = data[data.y.between(*data.y.quantile([0.05, 0.95]).tolist())] # TODO this removes some dates... may need to alter

        lag_start = 6
        lag_stop = 20

        for i in range(lag_start, lag_stop):
            data["lag_{}".format(i)] = data.y.shift(i)

        data['year'] = day_input['year']
        data['month'] = day_input['month']
        data['day'] = day_input['day']
        data["weekday"] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5,6])*1
        data['transaction_count'] = day_input['transaction_count']
        data['avg_grp_size'] = day_input['avg_grp_size']

        tscv = TimeSeriesSplit(n_splits=5) # TODO dynamic split selection

        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
        # used to transform into same scale given features vary in range
        scaler = StandardScaler()

        # necessary to handle sparsity; TODO move fucntion up stream to improve performance
        if (X_train.shape[0] <= 2) or (X_test.shape[0] <= 2):
            prediction_dates = y_test.index
            revenue_prediction = [None] * X_test.shape[0]
            
            output = pd.DataFrame({
                        "run_date": date.today(),
                        "prediction_dates": prediction_dates,
                        "est_total_revenue": revenue_prediction
                        })

            output_dict[dim] = output
            next
        
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            xgb = XGBRegressor()
            xgb.fit(X_train_scaled, y_train)

            predictions = xgb.predict(X_test_scaled)

            prediction_dates = y_test.index
            revenue_prediction = predictions

            output = pd.DataFrame({
                            "run_date": date.today(),
                            "prediction_dates": prediction_dates,
                            "est_total_revenue": revenue_prediction
                            })

            output_dict[dim] = output

    return output_dict

# COMMAND ----------

def reduce_transactions(transaction_df, interval):
    '''
    reduces length of transaction_df by interval in days 
    sums spend and counts transactions by customer for length of 
    (transaction_df minus interval = sample)
    interval in days will equal length of prediction
    '''
    start_date = transaction_df['DATE_D'].min() 
    end_date = transaction_df['DATE_D'].max() - timedelta(days=interval)
    
    reduced_transactions = transaction_df.loc[(transaction_df['DATE_D']>=start_date) &
                                              (transaction_df['DATE_D']<=end_date)]

    return reduced_transactions

# COMMAND ----------

def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)
  
  # Read single parquet file from S3
def pd_read_s3_parquet(key, bucket, s3_client=None, **args):
    if s3_client is None:
        s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()), **args)

# Read multiple parquets from a folder on S3 generated by spark
def pd_read_s3_multiple_parquets(filepath, bucket, s3=None, 
                                 s3_client=None, verbose=False, **args):
    if not filepath.endswith('/'):
        filepath = filepath + '/'  # Add '/' to the end
    if s3_client is None:
        s3_client = boto3.client('s3')
    if s3 is None:
        s3 = boto3.resource('s3')
    s3_keys = [item.key for item in s3.Bucket(bucket).objects.filter(Prefix=filepath)
               if item.key.endswith('.parquet')]
    if not s3_keys:
        print('No parquet found in', bucket, filepath)
    elif verbose:
        print('Load parquets:')
        for p in s3_keys: 
            print(p)
    dfs = [pd_read_s3_parquet(key, bucket=bucket, s3_client=s3_client, **args) 
           for key in s3_keys]
    return pd.concat(dfs, ignore_index=True)

# COMMAND ----------

transactions_pdf = pd_read_s3_multiple_parquets('apollo/alsea-demo/data/raw/facts/vip_txn_dtl_a_2019-01-01-present', 'cgp-data-lake')

# COMMAND ----------

transactions_pdf['date_time'] = pd.to_datetime(transactions_pdf['DATE_D'].apply(str)+' '+transactions_pdf['TIME_D'])

# COMMAND ----------

transactions_pdf.info()

# COMMAND ----------

# df = reduce_transactions(df)

# COMMAND ----------

df = create_date_features(transactions_pdf)

# COMMAND ----------

df.info()

# COMMAND ----------

all_input = format_agg_forecast_input(df)

product_input = format_product_forecast_input(df)
product_list = product_input.product_id.unique()

store_input = format_store_forecast_input(df)
store_list = store_input.store_id.unique()

# COMMAND ----------

'''
the four models below have been tested on the data sets
currently only one is selected in the creat_timeseries_output function
swap out the model or build additional ensemble logic with relative ease
'''
# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)

# ridge = RidgeCV(cv=tscv)
# ridge.fit(X_train_scaled, y_train)

# lasso = LassoCV(cv=tscv)
# lasso.fit(X_train_scaled, y_train)

# xgb = XGBRegressor()
# xgb.fit(X_train_scaled, y_train)
# ^^^ CURRENT MODEL SELECTED

# COMMAND ----------

product_forecasts = create_product_timeseries_output(product_list, product_input)

# adding the key in 
for key in product_forecasts.keys():
    product_forecasts[key]['key'] = key 

# concatenating the DataFrames
final_product_forecast_df = pd.concat(product_forecasts.values())

output = spark.createDataFrame(final_product_forecast_df)

# COMMAND ----------

final_product_forecast_df.head()

# COMMAND ----------

output.write.format('csv').save('s3://cgp-data-lake/apollo/alsea-demo/data/output/vips_product_level_forecasts')

# COMMAND ----------

store_forecasts = create_store_timeseries_output(store_list, store_input)

# adding the key in 
for key in store_forecasts.keys():
    store_forecasts[key]['key'] = key 

# concatenating the DataFrames
final_store_forecast_df = pd.concat(store_forecasts.values())

output = spark.createDataFrame(final_product_forecast_df)

# COMMAND ----------

final_store_forecast_df.head()

# COMMAND ----------

output.write.format('csv').save('s3://cgp-data-lake/apollo/alsea-demo/data/output/vips_store_level_forecasts')

# COMMAND ----------

agg_forecasts_df = create_aggregate_timeseries_output(all_input)

output = spark.createDataFrame(final_product_forecast_df)

# COMMAND ----------

agg_forecasts_df.head()

# COMMAND ----------

output.write.format('csv').save('s3://cgp-data-lake/apollo/alsea-demo/data/output/vips_agg_level_forecasts')