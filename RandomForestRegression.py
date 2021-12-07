from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType


def prepare_data():
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
    featurized = hasher.setHandleInvalid("skip").transform(data).select("features", "quality").withColumnRenamed(
        "quality",
        "label")

    return featurized


def Random_forest_Regression(train, crossValidation=False):
    rf = RandomForestRegressor(featuresCol="features", maxDepth=30, maxBins=40, numTrees=40)
    rfparamGrid = (ParamGridBuilder()
                   # .addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
                   .addGrid(rf.maxDepth, [10, 15])
                   # .addGrid(rf.maxBins, [10, 20, 40, 80, 100])
                   .addGrid(rf.maxBins, [20, 30, 40])
                   # .addGrid(rf.numTrees, [5, 20, 50, 100, 500])
                   .addGrid(rf.numTrees, [50, 60, 70])
                   .build())

    # Create 5-fold CrossValidator
    rfcv = CrossValidator(estimator=rf,
                          estimatorParamMaps=rfparamGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)

    # Run cross validations.
    if crossValidation:
        rfModel = rfcv.fit(train)
        print(rfModel.bestModel._java_obj.getMaxDepth())
        print(rfModel.bestModel._java_obj.getNumTrees())
    else:
        rfModel = rf.fit(train)
    return rfModel


def root_mean_error(predictions):
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


if __name__ == "__main__":
    featurized = prepare_data()
    train, test = featurized.randomSplit([0.8, 0.2], seed=12345)
    model = Random_forest_Regression(train)
    predictions = model.transform(test)
    predictions.select("prediction", "label", "features").show(5)
    # predictions.select("predictedLabel", "quality", "features").groupBy("predictedLabel").count().show()

    # Instantiate metrics object
    root_mean_error(predictions)
