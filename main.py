from __future__ import print_function
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window

from pyspark import SparkContext
from pyspark.sql import SparkSession, Window, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *

if __name__ == "__main__":
    # sc = SparkContext(appName="Pspark mllib Example")
    #
    # spark = SparkSession(sc)
    spark = SparkSession.builder.master("local[1]")\
                            .appName("SparkByExamples.com")\
                            .getOrCreate()
    data = spark.read.option("header",True).option("delimiter",";").csv("winequality-white.csv")
    # train, test = data.randomSplit([80, 20])
    data.printSchema()
    data.select(col('fixed acidity')).show()
    # print(data.show())
    # datat = sc.textFile("winequality-white.csv")
    # header = datat.first()
    # head = str(header).split(";")
    # data = datat.filter(lambda row: row != header)\
    #     # .map(lambda x: x.split(";")).toDF(head)
    # train, test = data.randomSplit([80, 20])
    # train = train.map(lambda x: x.split(";")).toDF(head)
    # test = test.map(lambda x: x.split(";")).toDF(head)
    # train.printSchema()
    # train.describe('alcohol').show()

    exit(0)
    #
    # print(train.count(), test.count(), data.count())
    # exit(0)
    # ratings = data.map(lambda l: l.split(';')) \
    #     .map(lambda l: Rating(float(l[0]), float(l[1]), float(l[2])))
    #
    #
    # # Build the recommendation model using Alternating Least Squares
    # rank = 10
    # numIterations = 10
    # model = ALS.train(ratings, rank, numIterations)
    #
    # # Evaluate the model on training data
    # testdata = ratings.map(lambda p: (p[0], p[1]))
    # predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    # ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    # MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    # print("Mean Squared Error = " + str(MSE))
    #
    # # Save and load model
    # model.save(sc, "target/tmp/myCollaborativeFilter")
    # sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")




