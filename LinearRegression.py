from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType


def prepare_data():
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
    data = spark.read.schema(SCHEMA).option("header", True).option("delimiter", ";") \
        .csv("winequality-white.csv")

    hasher = FeatureHasher(inputCols=[c for c in data.columns if c not in {'quality'}],
                           outputCol="features")
    featurized = hasher.transform(data).withColumnRenamed("quality", "label")
    return featurized


def Linear_Regression(train, crossValidation=False):
    lr = LinearRegression(maxIter=10,regParam=0.1,elasticNetParam=0.01)

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.5]) \
        .addGrid(lr.fitIntercept, [False, True]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    if(crossValidation):
        tvs = CrossValidator(estimator=lr,
                             estimatorParamMaps=paramGrid,
                             evaluator=RegressionEvaluator(),
                             numFolds=5)
        lrModel = tvs.fit(train)
        print(lrModel.bestModel._java_obj.getRegParam())
        print(lrModel.bestModel._java_obj.getElasticNetParam())

    else:
        lrModel= lr.fit(train)

    # Run TrainValidationSplit, and choose the best set of parameters.

    return lrModel


def root_mean_error(predictions):
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


if __name__ == "__main__":
    featurized = prepare_data()
    train, test = featurized.randomSplit([0.8, 0.2], seed=12345)

    model = Linear_Regression(train)
    predictions = model.transform(test)
    predictionAndLabels = predictions.select("prediction", "label")
    predictionAndLabels.show(5)

    root_mean_error(predictions)
