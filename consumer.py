from __future__ import print_function

import sys

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kafka_wordcount.py <zk> <topic>", file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="PythonStreamingKafka")
    spark = SparkSession(sc)
    ssc = StreamingContext(sc, 1)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

    def process_stream(record, spark):
        if not record.isEmpty():
            df = spark.createDataFrame(record)
            df.show()
            df.write.parquet("data_fromstream.parquet",mode="append")

    def parse(lp):
        print("///**"*80)
        print(lp)
        print("--||"*80)
        lp = lp[1]
        label = float(lp[lp.find('(') + 1: lp.find(')')])
        vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
        return LabeledPoint(label, vec)

    zkQuorum, topic = sys.argv[1:]
    kvs = KafkaUtils.createStream(ssc, zkQuorum, "spark-streaming-consumer", {topic: 1})
    #lines = kvs.map(lambda x: x[1]).map(parse)
    lines = kvs.map(parse)
    lines.foreachRDD(lambda rdd: process_stream(rdd, spark))

    lines.pprint()

    ssc.start()
    ssc.awaitTermination()
