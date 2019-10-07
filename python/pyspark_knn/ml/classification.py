from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaTransformer
from pyspark.ml.util import JavaMLWriter, JavaMLReader, JavaMLWritable, JavaMLReadable
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only

from pyspark_knn.ml.java_reader import CustomJavaMLReader


@inherit_doc
class KNNClassifier(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol,
                    HasProbabilityCol, HasRawPredictionCol, HasInputCols,
                    HasThresholds, HasSeed, HasWeightCol):
    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 seed=None, topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                 bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                 k=5, neighborsCol="neighbors", maxNeighbors=float("inf"), rawPredictionCol="rawPrediction",
                 probabilityCol="probability"):
        super(KNNClassifier, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.classification.KNNClassifier", self.uid)

        self.topTreeSize = Param(self, "topTreeSize", "number of points to sample for top-level tree")
        self.topTreeLeafSize = Param(self, "topTreeLeafSize",
                                     "number of points at which to switch to brute-force for top-level tree")
        self.subTreeLeafSize = Param(self, "subTreeLeafSize",
                                     "number of points at which to switch to brute-force for distributed sub-trees")
        self.bufferSize = Param(self, "bufferSize",
                                "size of buffer used to construct spill trees and top-level tree search")
        self.bufferSizeSampleSize = Param(self, "bufferSizeSampleSize",
                                          "number of sample sizes to take when estimating buffer size")
        self.balanceThreshold = Param(self, "balanceThreshold",
                                      "fraction of total points at which spill tree reverts back to metric tree if "
                                      "either child contains more points")
        self.k = Param(self, "k", "number of neighbors to find")
        self.neighborsCol = Param(self, "neighborsCol", "column names for returned neighbors")
        self.maxNeighbors = Param(self, "maxNeighbors", "maximum distance to find neighbors")

        self._setDefault(topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                         bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                         k=5, neighborsCol="neighbors", maxNeighbors=float("inf"))

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                  seed=None, topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                  bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                  k=5, neighborsCol="neighbors", maxNeighbors=float("inf"), rawPredictionCol="rawPrediction",
                  probabilityCol="probability"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return KNNClassificationModel(java_model)

    def setLabelCol(self, to):
        return self._set(labelCol=to)._call_java("setLabelCol", to)

    def setWeightCol(self, to):
        return self._set(weight_col=to)._call_java("setWeightCol", to)
        
    @property
    def inputCols(self):
        """
        """
        return self._call_java("fetchInputCols")

class KNNClassificationModel(JavaModel, JavaMLReadable, JavaMLWritable, HasFeaturesCol, HasLabelCol, HasPredictionCol,
HasProbabilityCol, HasRawPredictionCol, HasInputCols, HasThresholds, HasWeightCol):
    """
    Model fitted by KNNClassifier.
    """

    _classpath_model = 'org.apache.spark.ml.classification.KNNClassificationModel'

    def __init__(self, java_model):
        super(KNNClassificationModel, self).__init__(java_model)

        # note: look at https://issues.apache.org/jira/browse/SPARK-10931 in the future
        self.bufferSize = Param(self, "bufferSize", "size of buffer used to construct spill trees and top-level tree search")
        self.k = Param(self, "k", "number of neighbors to find")
        self.neighborsCol = Param(self, "neighborsCol", "column names for returned neighbors")
        self.distanceCol = Param(self, "distanceCol", "column name for computed distances")
        self.maxNeighbors = Param(self, "maxNeighbors", "maximum distance to find neighbors")

        self._resetUid(java_model.uid())

        self._transfer_params_from_java()

    @property
    def numClasses(self):
        """
        """
        return self._call_java("numClasses")
        
    @classmethod
    def _from_java(cls, java_stage):
        """
        Given Java KNNClassificationModel, create and return Python wrapper. Necessary for persistence.
        """
        py_type = KNNClassificationModel
        py_stage = None
        if issubclass(py_type, JavaParams):
            # Load information from java_stage to the instance.
            py_stage = py_type(java_stage)

        return py_stage

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CustomJavaMLReader(cls, cls._classpath_model)