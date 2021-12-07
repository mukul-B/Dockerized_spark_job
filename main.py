from pyspark.ml.feature import FeatureHasher
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator



# spark = SparkSession.builder.master("local[1]") \
#     .appName("cch2") \
#     .getOrCreate()
#
# SCHEMA = StructType([StructField('fixed acidity', DoubleType()),
#                      StructField('volatile acidity', DoubleType()),
#                      StructField('citric acid', DoubleType()),
#                      StructField('residual sugar', DoubleType()),
#                      StructField('chlorides', DoubleType()),
#                      StructField('free sulfur dioxide', IntegerType()),
#                      StructField('total sulfur dioxide', IntegerType()),
#                      StructField('density', DoubleType()),
#                      StructField('pH', DoubleType()),
#                      StructField('sulphates', DoubleType()),
#                      StructField('alcohol', DoubleType()),
#                      StructField('quality', IntegerType())])
# # Prepare training and test data.
# data = spark.read.schema(SCHEMA).option("header", True).option("delimiter", ";") \
#     .csv("winequality-white.csv")
#
# hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
#                        outputCol="features")
# featurized = hasher.transform(data)
# # featurized.show(truncate=False)
# train, test = featurized.randomSplit([0.8, 0.2], seed=12345)
#
# lr = LinearRegression(maxIter=20, regParam=0.3, elasticNetParam=0.8).setLabelCol("quality")
# lrModel = lr.fit(train)
#
# # lr = RandomForestRegressor(numTrees=2, maxDepth=2)
# # lrModel = lr.fit(train)
#
# # print("Coefficients: %s" % str(lrModel.coefficients))
# # print("Intercept: %s" % str(lrModel.intercept))
#
# predictions = lrModel.transform(test)
# predictionAndLabels=predictions.select("prediction", "quality")
# predictionAndLabels.show(5)
#
# evaluator = RegressionEvaluator(
#     labelCol="quality", predictionCol="prediction", metricName="rmse")
#
# rmse = evaluator.evaluate(predictions)
# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
#
# # evaluatorMulti = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
# #
# # f1 = evaluatorMulti.evaluate(predictionAndLabels, {evaluatorMulti.metricName: "f1"})
# #
# # print(f1)


