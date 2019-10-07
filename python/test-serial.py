import logging

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.linalg import Vectors
from pyspark_knn.ml.classification import KNNClassifier, KNNClassificationModel

# This is a simple test app. Use the following command to run:
# $SPARK_HOME/bin/spark-submit --py-files python/dist/pyspark_knn-*.egg --driver-class-path spark-knn-core/target/scala-2.11/spark-knn_*.jar --jars spark-knn-core/target/scala-2.11/spark-knn_*.jar python/test.py

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)s] %(message)s")
logger = logging.getLogger("spark-knn-test")

spark = SparkSession.builder.appName("spark-knn-test").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

logger.info("Alive.")

training = sqlContext.createDataFrame([
    [Vectors.sparse(3, [0, 2], [1.0, 3.0]), 0.0],
    [Vectors.sparse(3, [0, 1], [-1.0, -2.0]), 1.0],
    [Vectors.sparse(3, [2], [2.0]), 0.0],
    [Vectors.sparse(3, [1, 2], [-1.0, 1.0]), 1.0],
], ['mike', 'easter'])

test = sqlContext.createDataFrame([
    [Vectors.sparse(3, [0, 1, 2], [0.6, 0.1, 0.9])],        # s/b 0.0
    [Vectors.sparse(3, [0, 1, 2], [-0.3, -0.8, 0.2])]       # s/b 1.0
], ['mike'])

knn = KNNClassifier(k=1, topTreeSize=1, topTreeLeafSize=1, subTreeLeafSize=1, bufferSizeSampleSize=[1, 2, 3])  # bufferSize=-1.0,
logger.info('Param names: {}'.format([p.name for p in knn.params]))

logger.info("Estimator k: {}".format(knn.getOrDefault(knn.getParam("k"))))
logger.info("Estimator uid: {}".format(knn.uid))
logger.info("Estimator cols (features, label, neighbors): '{}', '{}', '{}'".format(knn.getFeaturesCol(), knn.getLabelCol(), knn.getOrDefault(knn.getParam("neighborsCol"))))
knn.setFeaturesCol("mike")
knn.setLabelCol("easter")
logger.info("Estimator cols (features, label, neighbors, input): '{}', '{}', '{}', '{}'"
    .format(knn.getFeaturesCol(), knn.getLabelCol(), knn.getOrDefault(knn.getParam("neighborsCol")), knn.inputCols))

logger.info('Fitting.')

model = knn.fit(training)

logger.info("Model uid: {}".format(model.uid))
logger.info("Model numClasses: {}".format(model.numClasses))
logger.info("Model cols (features, label, neighbors): {}, {}, {}".format(model.getFeaturesCol(), model.getLabelCol(), model.neighborsCol))

logging.info('Predicting.')
predictions = model.transform(test)
logger.info('Predictions:\n\t{}'.format('\n\t'.join(["{:.1f}".format(i.prediction) for i in predictions.collect()])))

model_path = 'hdfs:///tmp/knn-save-attempt'
logger.info("Attempting save.")
model.write().overwrite().save(model_path)
logger.info("Model saved to '{}'.".format(model_path))

logger.info("Attempting load.")
reconstituted = KNNClassificationModel.load(model_path)
logger.info("Model loaded from '{}', uid {}.".format(model_path, reconstituted.uid))
logger.info('Transformer params: {}'.format([p.name for p in reconstituted.params]))
logger.info("Transformer numClasses: {}".format(reconstituted.numClasses))
logger.info("Transformer k: {}".format(reconstituted.getOrDefault(reconstituted.getParam("k"))))
logger.info("Reconstituted cols (features, label, neighbors): {}, {}, {}"
    .format(reconstituted.getFeaturesCol(), reconstituted.getLabelCol(), reconstituted.getOrDefault(reconstituted.neighborsCol)))

logging.info('Predicting on reconstituted.')
predictions = reconstituted.transform(test)
logger.info('Predictions on reconstituted:\n\t{}'.format('\n\t'.join(["{:.1f}".format(i.prediction) for i in predictions.collect()])))