from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import VectorAssembler
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

    hasher = VectorAssembler(inputCols=[c for c in data.columns if c not in {'quality'}],
                             outputCol="features")
    featurized = hasher.setHandleInvalid("skip").transform(data).select("features", "quality").withColumnRenamed(
        "quality", "label")

    return featurized


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


def precision_recall(predictions):
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # f1Score = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
    weightedPrecision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})

    # print("F1 Score = %s" % f1Score)
    print("weightedPrecision = %s" % weightedPrecision)
    print("weightedRecall =", weightedRecall)


if __name__ == "__main__":
    featurized = prepare_data()
    train, test = featurized.randomSplit([0.8, 0.2], seed=12345)
    model = Random_forest_classification(train)
    predictions = model.transform(test)
    predictions.select("prediction", "label", "features").show(5)
    # predictions.select("prediction", "label", "features").groupBy("predictedLabel").count().show()
    precision_recall(predictions)
