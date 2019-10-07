from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.linalg import Vectors
from pyspark_knn.ml.classification import KNNClassifier

# This is a simple test app. Use the following command to run:
# $SPARK_HOME/bin/spark-submit --py-files python/dist/pyspark_knn-*.egg --driver-class-path spark-knn-core/target/scala-2.11/spark-knn_*.jar --jars spark-knn-core/target/scala-2.11/spark-knn_*.jar python/test.py

spark = SparkSession.builder.appName("Spark-knn-test").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

print('Initializing')
training = sqlContext.createDataFrame([
    [Vectors.sparse(3, [0, 2], [1.0, 3.0]), 0.0],
    [Vectors.sparse(3, [0, 1], [-1.0, -2.0]), 1.0],
    [Vectors.sparse(3, [2], [2.0]), 0.0],
    [Vectors.sparse(3, [1, 2], [-1.0, 1.0]), 1.0],
], ['features', 'label'])

test = sqlContext.createDataFrame([
    [Vectors.sparse(3, [0, 1, 2], [0.6, 0.1, 0.9])],        # s/b 0.0
    [Vectors.sparse(3, [0, 1, 2], [-0.3, -0.8, 0.2])]       # s/b 1.0
], ['features'])

knn = KNNClassifier(k=1, topTreeSize=1, topTreeLeafSize=1, subTreeLeafSize=1, bufferSizeSampleSize=[1, 2, 3])  # bufferSize=-1.0,
print('Params:', [p.name for p in knn.params])
print('Fitting')
model = knn.fit(training)
print('bufferSize:', model._java_obj.getBufferSize())
print('Predicting')
predictions = model.transform(test)
print('Predictions:')
for row in predictions.collect():
    print(row)
