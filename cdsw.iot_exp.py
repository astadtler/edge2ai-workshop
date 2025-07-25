#from pyspark.sql import SparkSession
from snowflake.snowpark import Session
#from pyspark.sql.types import *
from snowflake.snowpark.types import StructType, StructField, DoubleType, IntegerType
#from pyspark.ml.feature import StringIndexer
from snowflake.ml.modeling.preprocessing import
#from pyspark.ml import Pipeline
from snowflake.ml.modeling.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier
from snowflake.ml.modeling.ensemble import RandomForestClassifier
#from sklearn.metrics import roc_auc_score, average_precision_score
from snowflake.ml.modeling.metrics import roc_auc_score, precision_score

import numpy as np
import pandas as pd
import pickle
#import cdsw
from snowflake.ml.registry import Registry
import os
import time

session = Session.builder.config("connection_name", "myconnection").create()

# read 21 colunms large file from Snowflake stage
schemaData = StructType([StructField("C1", DoubleType(), True),
                     StructField("C2", DoubleType(), True),
                     StructField("C3", DoubleType(), True),
                     StructField("C4", DoubleType(), True),
                     StructField("C5", DoubleType(), True),
                     StructField("C6", DoubleType(), True),
                     StructField("C7", DoubleType(), True),
                     StructField("C8", DoubleType(), True),
                     StructField("C9", DoubleType(), True),
                     StructField("C10", DoubleType(), True),
                     StructField("C11", DoubleType(), True),
                     StructField("C12", DoubleType(), True),
                     StructField("C13", IntegerType(), True)])

iot_data = session.read.schema(schemaData).csv('@edge2ai/historical_iot.txt')

# Create Pipeline
label_indexer = LabelEncoder(input_cols = ["C13"], output_cols = ["LABEL"])
plan_indexer = LabelEncoder(input_cols = ["C2"], output_cols = ["C2_INDEXED"])
pipeline = Pipeline(steps=[('plan_indexer', plan_indexer), ('label_indexer', label_indexer)])
indexed_data = pipeline.fit(iot_data).transform(iot_data)
(train_data, test_data) = indexed_data.random_split([0.7, 0.3])

#pdTrain = train_data.toPandas()
#pdTest = test_data.toPandas()

# 12 features
features = ["C2_INDEXED",
            "C1",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
            "C10",
            "C11",
            "C12"]

param_numTrees = int(sys.argv[1])
param_maxDepth = int(sys.argv[2])
param_impurity = 'gini'

randF = RandomForestClassifier(n_jobs=10,
                               n_estimators=param_numTrees,
                               max_depth=param_maxDepth,
                               criterion=param_impurity,
                               random_state=0,
                               input_cols=features,
                               label_cols=['LABEL'],
                               output_cols=['PREDICTIONS'])
                               )

model.set_metric(metric_name="numTrees",value=param_numTrees)
model.set_metric(metric_name="maxDepth",value=param_maxDepth)
model.set_metric(metric_name="impurity",value=param_impurity)

# Fit and Predict
randF.fit(train_data)
predictions=randF.predict(test_data)

#log model
registry = Registry(session=session)

model = registry.log_model(
    model_name="EDGE2AI_RANDOM_FOREST",
    model=randF,
    sample_input_data=test_data, # to provide the feature schema
    target_platforms={'WAREHOUSE'})

#temp = randF.predict_proba(pdTest[features])

pd.crosstab(pdTest['label'], predictions, rownames=['Actual'], colnames=['Prediction'])

list(zip(pdTrain[features], randF.feature_importances_))


y_true = predictions['LABEL']
y_scores = predictions['PREDICTIONS']
auroc = roc_auc_score(df=predictions, y_true_col_names=['LABEL'], y_score_col_names=['PREDICTIONS'])
ap = precision_score (df=predictions, y_true_col_names=['LABEL'], y_pred_col_names=['PREDICTIONS'])
print(auroc, ap)


#cdsw.track_metric("auroc", auroc)
model.set_metric(metric_name="auroc", value=auroc)
model.set_metric(metric_name="ap", value=ap)
#cdsw.track_metric("ap", ap)

#pickle.dump(randF, open("iot_model.pkl","wb"))

#cdsw.track_file("iot_model.pkl")

#Create SPCS Inference Service
model.create_service(service_name="EDGE2AI",
                     service_compute_pool="EDGE2AI",
                     image_repo="EDGE2AI_REPO",
                     ingress_enabled=True,
                     gpu_requests=None)
time.sleep(15)
print("Slept for 15 seconds.")
