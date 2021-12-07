from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.feature import VectorAssembler
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

hasher = VectorAssembler(inputCols=[c for c in data.columns if c not in {'quality'}],
                         outputCol="features")
featurized = hasher.setHandleInvalid("skip").transform(data).select("features", "quality").withColumnRenamed("quality", "label")

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(featurized)

train, test = featurized.randomSplit([0.8, 0.2], seed=12345)

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)


labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])
rfModel = pipeline.fit(train)

predictions = rfModel.transform(test)
predictions.select("predictedLabel", "label", "features").show(15)

# predictions.select("predictedLabel", "quality", "features").groupBy("predictedLabel").count().show()

# Instantiate metrics object

def precision_recall(predictions):
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # f1Score = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
    # weightedPrecision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})

    # print("F1 Score = %s" % f1Score)
    # print("weightedPrecision = %s" % weightedPrecision)
    print("weightedRecall =", weightedRecall)


precision_recall(predictions)