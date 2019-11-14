import logging

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf, col, array_contains, when
from pyspark.sql.types import *
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

tr_path = "hdfs:///user/dtracy/rekt/out/unit/features/training"
val_path = "hdfs:///user/dtracy/rekt/out/unit/features/validation/split"
model_path = "hdfs:///tmp/knn-save-attempt"
active_class_id = 7.0

knn = KNNClassifier(k=25, 
                    topTreeSize=1000, 
                    topTreeLeafSize=10, 
                    subTreeLeafSize=30, 
                    bufferSizeSampleSize=list(range(100, 1000 + 1, 100)),
                    featuresCol='combined__f',
                    labelCol='active_category_index',
                    weightCol='weights')

def label_one(cats_col, cat):
    res = 0.0
    if cat in cats_col:
        res = 1.0
    return res

def label_one_factory(cat):
    return udf(lambda x: label_one(x, cat), DoubleType())

f_df = spark.read.format('parquet') \
    .load(tr_path) \
    .repartition(10) \
    .cache()
f_count = f_df.count()

fprime_df = f_df.withColumn('active_category_index', label_one_factory(active_class_id)('__indices'))
pos_count = fprime_df.where(col('active_category_index') == 1.0).count()
pos_weight = (f_count/pos_count) * 0.5
fprime_df = fprime_df.withColumn('weights', when(fprime_df['active_category_index'] == 1.0, pos_weight).otherwise(1.0))

model = knn.fit(fprime_df)

model.write().overwrite().save(model_path)

reconstituted = KNNClassificationModel.load(model_path)
logger.info("Model loaded from '{}', uid {}.".format(model_path, reconstituted.uid))
logger.info('Transformer params: {}'.format([p.name for p in reconstituted.params]))
logger.info("Transformer numClasses: {}".format(reconstituted.numClasses))
logger.info("Transformer k: {}".format(reconstituted.getOrDefault(reconstituted.getParam("k"))))
logger.info("Reconstituted cols (features, label, input): {}, {}, {}"
    .format(reconstituted.getFeaturesCol(), reconstituted.getLabelCol(), reconstituted.getInputCols()))

test = spark.read.format('parquet') \
    .load(val_path) \
    .repartition(10) \
    .cache()

logging.info('Predicting with reconstituted.')
predictions = reconstituted.transform(test)
logger.info("All predictions columns: {}".format(predictions.columns))

logger.info('Predictions with reconstituted:\n\t{}'.format(predictions.head(10)))

pred_tally_df = predictions.groupBy('prediction').count()

logger.info('Predictions breakdown:\n')
pred_tally_df.show()

logger.info('Rows with positive predictions:')
predictions.where(col('prediction') == 1.0).show()

pos_df = fprime_df.where(array_contains('__indices', active_class_id))
logger.info('Rows with the active category id in training ({} rows):'.format(pos_df.count()))
pos_df.show()

potential_df = predictions.where(array_contains('__indices', active_class_id))
logger.info('Rows with the active category id in predictions ({} rows):'.format(potential_df.count()))
potential_df.show()