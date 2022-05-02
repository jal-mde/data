from __future__ import print_function

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
# $example on$
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.mllib.clustering import StreamingKMeansModel
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="StreamingKMeansExample")  # SparkContext
    ssc = StreamingContext(sc, 1)

    # $example on$
    # we make an input stream of vectors for training,
    # as well as a stream of vectors for testing
    def parse(lp):
        label = float(lp[lp.find('(') + 1: lp.find(')')])
        vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))

        return LabeledPoint(label, vec)

    trainingData = sc.textFile("train.csv")\
        .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(',')]))

    testingData = sc.textFile("test.csv").map(parse)

    trainingQueue = [trainingData]
    testingQueue = [testingData]

    trainingStream = ssc.queueStream(trainingQueue)
    testingStream = ssc.queueStream(testingQueue)

    # We create a model with random clusters and specify the number of clusters to find
    model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(6, 1.0, 0)

    # Now register the streams for training and testing and start the job,
    # printing the predicted cluster assignments on new data points as they arrive.
    model.trainOn(trainingStream)

    result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
    result.pprint()

    ssc.start()
    ssc.stop(stopSparkContext=True, stopGraceFully=True)
    # $example off$

    print("Final centers: " + str(model.latestModel().centers))
