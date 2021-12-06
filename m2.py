from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import FeatureHasher, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.master("local[1]") \
    .appName("cch2") \
    .getOrCreate()

SCHEMA = StructType([StructField('fixed acidity', DoubleType()),
                     StructField('volatile acidity', DoubleType()),
                     StructField('citric acid', DoubleType()),
                     StructField('residual sugar', DoubleType()),
                     StructField('chlorides', DoubleType()),
                     StructField('free sulfur dioxide', IntegerType()),
                     StructField('total sulfur dioxide', IntegerType()),
                     StructField('density', DoubleType()),
                     StructField('pH', DoubleType()),
                     StructField('sulphates', DoubleType()),
                     StructField('alcohol', DoubleType()),
                     StructField('quality', IntegerType())])
# Prepare training and test data.
#     .config("spark.driver.memory", "15g")\
data = spark.read.schema(SCHEMA).option("header", True).option("delimiter", ";") \
    .csv("winequality-white.csv")
# data, left, = datao.randomSplit([0.01, 0.99], seed=12345)

# print(data.count())
# exit(0)
hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
                       outputCol="features")
featurized = hasher.transform(data).select("features","quality")
# featurized.show(truncate=False)
# exit(0)
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(featurized)
labelIndexer = StringIndexer(inputCol="quality", outputCol="indexedLabel").fit(data)
train, test, = featurized.randomSplit([0.8, 0.2], seed=12345)

# print(featurized.count(),test.count(),train.count(),left.count())

# rf = RandomForestRegressor(featuresCol="indexedFeatures").setLabelCol("quality")
#
#
# # Chain indexer and forest in a Pipeline
# pipeline = Pipeline(stages=[featureIndexer, rf])

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

rfModel = pipeline.fit(train)

# Make predictions.
predictions = rfModel.transform(test)
# Select example rows to display.
predictions.select("prediction", "quality").show(5)
#
# # Select (prediction, true label) and compute test error
# evaluator = RegressionEvaluator(
#     labelCol="label", predictionCol="prediction", metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
#
# rfModel = rfModel.stages[1]
# print(rfModel)  # summary only


