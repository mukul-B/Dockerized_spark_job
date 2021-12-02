from pyspark.ml.feature import FeatureHasher
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
    .config("spark.driver.memory", "15g")\
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
data = spark.read.schema(SCHEMA).option("header", True).option("delimiter", ";") \
    .csv("winequality-white.csv")

hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
                       outputCol="features")
featurized = hasher.transform(data).select("features","quality")
# featurized.show(truncate=False)
# exit(0)
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(featurized)

train, test = featurized.randomSplit([0.8, 0.2], seed=12345)

rf = RandomForestRegressor(featuresCol="indexedFeatures").setLabelCol("quality")


# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

rfModel = pipeline.fit(train)

# Make predictions.
predictions = rfModel.transform(test)
# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = rfModel.stages[1]
print(rfModel)  # summary only


