# data
# Databricks notebook source
# data management and transformation
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import glob
from datetime import datetime as dt
from datetime import timedelta 
from datetime import date

# viz
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# prediction
import sklearn 
from sklearn.model_selection import train_test_split
from hpfrec import HPF
from sklearn.metrics import roc_auc_score

# COMMAND ----------

spark = SparkSession.builder.appName('product_rec_hpf').getOrCreate()

# COMMAND ----------

transactions = spark.read.option('inferSchema', True).option('header', True).format('csv').load('s3://cgp-data-lake/sandbox/tristan/alsea/test/*.csv')

# COMMAND ----------

transactions.createOrReplaceTempView('transactions')

# COMMAND ----------

product_counts = spark.sql('SELECT MEMBER_WID AS UserId, PRODUCT_WID AS ItemId, COUNT(PRODUCT_WID) AS Count FROM transactions GROUP BY MEMBER_WID, PRODUCT_WID').toPandas()

# COMMAND ----------

product_info = spark.read.format('csv').option('inferSchema', True).option('header', True).load('s3://cgp-data-lake/sandbox/tristan/alsea/dims/cat_prods.csv').toPandas()

# COMMAND ----------

train, test = train_test_split(product_counts, test_size=.25, random_state=1)

users_train = set(train.UserId)
items_train = set(train.ItemId)

test = test.loc[(test.UserId.isin(users_train)) & (test.ItemId.isin(items_train))].reset_index(drop=True)
del users_train, items_train
del product_counts
test.shape

# COMMAND ----------

## Full call would be like this:
# recommender = HPF(k=50, a=0.3, a_prime=0.3, b_prime=1.0,
#                  c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
#                  stop_crit='train-llk', check_every=10, stop_thr=1e-3,
#                  maxiter=150, reindex=True, random_seed = 123,
#                  allow_inconsistent_math=False, verbose=True, full_llk=False,
#                  keep_data=True, save_folder=None, produce_dicts=True)

# For more information see the documentation:
# http://hpfrec.readthedocs.io/en/latest/

recommender = HPF(k=50, full_llk=False, random_seed=123,
                  check_every=10, maxiter=150, reindex=True,
                  allow_inconsistent_math=True, ncores=24,
                  verbose=True,
                  save_folder='/tmp')

# COMMAND ----------

recommender.fit(train)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cat /tmp/hyperparameters.txt

# COMMAND ----------

# common sense checks 
test['Predicted'] = recommender.predict(user=test.UserId, item=test.ItemId)
test['RandomItem'] = np.random.choice(train.ItemId, size=test.shape[0])
test['PredictedRandom'] = recommender.predict(user=test.UserId, item=test.RandomItem)
print("Average prediction for combinations in test set: ", test.Predicted.mean())
print("Average prediction for random combinations: ", test.PredictedRandom.mean())

# COMMAND ----------

"""
As some common sense checks, the predictions should:
- Be higher for this non-zero hold-out sample than for random items
- Produce a good discrimination between random items and those in the hold-out sample (very related to the first point).
- Be correlated with the counts in the hold-out sample
- Follow an exponential distribution rather than a normal or some other symmetric distribution.
"""

was_purchased = np.r_[np.ones(test.shape[0]), np.zeros(test.shape[0])]
score_model = np.r_[test.Predicted.values, test.PredictedRandom.values]
roc_auc_score(was_purchased, score_model)

# COMMAND ----------

print('correlation of predictions to hold out sample. we want this to be close to 1')
np.corrcoef(test.Count, test.Predicted)[0,1]

# COMMAND ----------

# %matplotlib inline

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC # this output should follow an exponential distribution rather than a normal or some other symmetric distribution
# MAGIC 
# MAGIC _ = plt.hist(test.Predicted, bins=1000) # try alternate bin sizes 
# MAGIC plt.xlim(0,0.4) # adjust x axis range as needed
# MAGIC plt.show()

# COMMAND ----------

'''
Pick 3 random users with a reasonably long history of purchases
Check which products exist in the training data with which the model was fit
See which products the model recommend to them among those which they have not yet purchased
Top-N lists can be made among all items, or across some user-provided subset only
'''
# %%
total_purchases_by_user = train.groupby('UserId').agg({'Count':np.sum})
total_purchases_by_user = total_purchases_by_user[total_purchases_by_user.Count > 3]

np.random.seed(1)
sample_users = np.random.choice(total_purchases_by_user.index, 3)

# COMMAND ----------

print('sample users')
sample_users

# COMMAND ----------

print('recommended products for sample user 1')
recommender.topN(user = sample_users[0],
                 n=3, exclude_seen = True) # n = product recommendations

# COMMAND ----------

print('examine individual recommendation details of sample users')
train_rec = pd.merge(train, product_info, left_on='ItemId', right_on='PROD_WID')
train_rec
# TODO potential improvement to model: build model on product line (PR_PROD_LN) then randomly 
# select item from group

# COMMAND ----------

print('top items purchased from sample customer #1')
x = train_rec.loc[train_rec.UserId==sample_users[0]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
x

# COMMAND ----------

print('recommendations for sample customer #1')
recommended_list = recommender.topN(sample_users[0], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# COMMAND ----------

print('top items purchased from sample customer #2')
y = train_rec.loc[train_rec.UserId==sample_users[1]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
y

# COMMAND ----------

print('recommendations for sample customer #2')
recommended_list = recommender.topN(sample_users[1], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# COMMAND ----------

print('top items purchased from sample customer #3')
y = train_rec.loc[train_rec.UserId==sample_users[2]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
y

# COMMAND ----------

print('recommendations for sample customer #3')
recommended_list = recommender.topN(sample_users[2], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# COMMAND ----------

# TODO notice WOW ticket recommendations present - wonder if outlier and should be removed
