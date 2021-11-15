from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, Window, Row
from pyspark.sql.functions import *

spark = SparkSession.builder.master("local[1]")\
                            .appName("SparkByExamples.com")\
                            .getOrCreate()
# Prepare training and test data.
data = spark.read.option("header",True).option("delimiter",";")\
    .csv("winequality-white.csv")
train, test = data.randomSplit([0.8, 0.2], seed=12345)
print(test.count())
print(train.count())
train.select(col('fixed acidity')).show()

