from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

spark = SparkSession.builder.master("local[1]") \
    .appName("cch2") \
    .config("spark.driver.memory", "15g") \
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

# hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
#                        outputCol="features")
# featurized = hasher.transform(data).select("features","quality").withColumnRenamed("quality", "label")
hasher = VectorAssembler(inputCols=[c for c in data.columns if c not in {'quality'}],
                         outputCol="features")
featurized = hasher.setHandleInvalid("skip").transform(data).select("features", "quality").withColumnRenamed("quality",
                                                                                                             "label")

train, test = featurized.randomSplit([0.8, 0.2], seed=12345)

rf = RandomForestRegressor(featuresCol="features", maxDepth=5,maxBins=20,numTrees=50)

pipeline = Pipeline(stages=[rf])

rfparamGrid = (ParamGridBuilder()
               .addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
               # .addGrid(rf.maxDepth, [2, 5, 10])
               .addGrid(rf.maxBins, [10, 20, 40, 80, 100])
               # .addGrid(rf.maxBins, [5, 10, 20])
               .addGrid(rf.numTrees, [5, 20, 50, 100, 500])
               # .addGrid(rf.numTrees, [5, 20, 50])
               .build())


# Create 5-fold CrossValidator
rfcv = CrossValidator(estimator=rf,
                      estimatorParamMaps=rfparamGrid,
                      evaluator=RegressionEvaluator(),
                      numFolds=5)

# print(rfcv.getParam("maxBins"))
# print(rfcv.getParam("numTrees"))
# Run cross validations.
rfModel = rfcv.fit(train)
print(rfModel.getEstimatorParamMaps())

# rfModel = pipeline.fit(train)

predictions = rfModel.transform(test)
predictions.select("prediction", "label", "features").show(5)
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
