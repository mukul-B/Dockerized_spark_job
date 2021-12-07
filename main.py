import sys

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType


def prepare_data(hash=False):
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

    if (hash):

        hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
                               outputCol="features")
        featurized = hasher.transform(data).withColumnRenamed("quality", "label")


    else:
        hasher = VectorAssembler(inputCols=[c for c in data.columns if c not in {'quality'}],
                                 outputCol="features")
        featurized = hasher.setHandleInvalid("skip").transform(data).select("features", "quality").withColumnRenamed(
            "quality", "label")

    return featurized


def Linear_Regression(train, crossValidation=False):
    lr = LinearRegression(maxIter=10, regParam=0.1, elasticNetParam=0.01)

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.5]) \
        .addGrid(lr.fitIntercept, [False, True]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    if (crossValidation):
        tvs = CrossValidator(estimator=lr,
                             estimatorParamMaps=paramGrid,
                             evaluator=RegressionEvaluator(),
                             numFolds=5)
        lrModel = tvs.fit(train)
        print(lrModel.bestModel._java_obj.getRegParam())
        print(lrModel.bestModel._java_obj.getElasticNetParam())

    else:
        lrModel = lr.fit(train)

    # Run TrainValidationSplit, and choose the best set of parameters.

    return lrModel


def Random_forest_classification(train, crossValidation=False):
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=25, numTrees=30)

    rfparamGrid = (ParamGridBuilder()
                   .addGrid(rf.maxDepth, [20, 25, 30])
                   .addGrid(rf.numTrees, [25, 30, 35])
                   .build())

    # Create 5-fold CrossValidator
    rfcv = CrossValidator(estimator=rf,
                          estimatorParamMaps=rfparamGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
    # Run cross validations.
    if crossValidation:
        rfModel = rfcv.fit(train)
        print(rfModel.bestModel._java_obj.getMaxDepth())
        print(rfModel.bestModel._java_obj.getNumTrees())
    else:
        rfModel = rf.fit(train)
    return rfModel


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


def precision_recall(predictions):
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # f1Score = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
    weightedPrecision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})

    # print("F1 Score = %s" % f1Score)
    print("weightedPrecision = %s" % weightedPrecision)
    print("weightedRecall =", weightedRecall)


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("one argument is required for model type, running random forest regression")
        model_type = "lr"
    else:
        model_type = sys.argv[1]
    print(model_type)
    if (model_type == "rr"):
        featurized = prepare_data()
        train, test = featurized.randomSplit([0.8, 0.2], seed=12345)
        model = Random_forest_Regression(train)
        predictions = model.transform(test)
        predictions.select("prediction", "label", "features").show(5)
        # predictions.select("predictedLabel", "quality", "features").groupBy("predictedLabel").count().show()

        # Instantiate metrics object
        root_mean_error(predictions)
    elif (model_type == "rc"):
        featurized = prepare_data()
        train, test = featurized.randomSplit([0.8, 0.2], seed=12345)
        model = Random_forest_classification(train)
        predictions = model.transform(test)
        predictions.select("prediction", "label", "features").show(5)
        # predictions.select("prediction", "label", "features").groupBy("predictedLabel").count().show()
        precision_recall(predictions)
    elif (model_type == "lr"):
        featurized = prepare_data(True)
        train, test = featurized.randomSplit([0.8, 0.2], seed=12345)

        model = Linear_Regression(train)
        predictions = model.transform(test)
        predictionAndLabels = predictions.select("prediction", "label")
        predictionAndLabels.show(5)

        root_mean_error(predictions)
    else:
        print("wrong argument : lr(linear regression) , rr(random forest regression) ,rc(random forest classifier)")
