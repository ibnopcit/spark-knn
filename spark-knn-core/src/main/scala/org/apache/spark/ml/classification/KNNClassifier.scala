package org.apache.spark.ml.classification

import java.io._

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j

import org.apache.spark.ml.classification.KNNClassificationModel.KNNClassificationModelWriter
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.knn._
import org.apache.spark.ml.knn.KNNModelParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.{Identifiable, SchemaUtils, MLReadable, MLWritable, MLReader, MLWriter}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.persistence.knn._
import org.apache.spark.persistence.knn.{FsagSerialization}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkException

import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, Formats}
import scala.collection.mutable.ArrayBuffer

/**
  * [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for classification.
  * An object is classified by a majority vote of its neighbors, with the object being assigned to
  * the class most common among its k nearest neighbors.
  */
class KNNClassifier(override val uid: String) extends ProbabilisticClassifier[Vector, KNNClassifier, KNNClassificationModel]
with KNNParams with HasWeightCol {

  def this() = this(Identifiable.randomUID("knnc"))

  def fetchInputCols: Array[String] = $(inputCols)

  /** @group setParam */
  override def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  override def setLabelCol(value: String): this.type = {
    set(labelCol, value)

    var rval: this.type = null
    if ($(weightCol).isEmpty) {
      rval = set(inputCols, Array(value))
    } else {
      rval = set(inputCols, Array(value, $(weightCol)))
    }
    rval
  }

  //fill in default label col
  setDefault(inputCols, Array($(labelCol)))

  /** @group setWeight */
  def setWeightCol(value: String): this.type = {
    set(weightCol, value)

    if (value.isEmpty) {
      set(inputCols, Array($(labelCol)))
    } else {
      set(inputCols, Array($(labelCol), value))
    }
  }

  setDefault(weightCol -> "")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

  /** @group setParam */
  def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

  /** @group setParam */
  def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

  /** @group setParam */
  def setBufferSizeSampleSizes(value: Array[Int]): this.type = set(bufferSizeSampleSizes, value)

  /** @group setParam */
  def setBalanceThreshold(value: Double): this.type = set(balanceThreshold, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  override protected def train(dataset: Dataset[_]): KNNClassificationModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSummarizer = instances.treeAggregate(
      new MultiClassSummarizer)(
      seqOp = (c, v) => (c, v) match {
        case (labelSummarizer: MultiClassSummarizer, (label: Double, features: Vector)) =>
          labelSummarizer.add(label)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case (classSummarizer1: MultiClassSummarizer, classSummarizer2: MultiClassSummarizer) =>
          classSummarizer1.merge(classSummarizer2)
      })

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length

    if (numInvalid != 0) {
      val msg = s"Classification labels should be in {0 to ${numClasses - 1} " +
        s"Found $numInvalid invalid labels."
      logError(msg)
      throw new SparkException(msg)
    }

    // KNN object deals with inputCols field only, so set its components before copying.
    val knnModel = copyValues(new KNN().setAuxCols($(inputCols))).fit(dataset)
    knnModel.toNewClassificationModel(uid, numClasses)
  }

  override def fit(dataset: Dataset[_]): KNNClassificationModel = {
    // Need to overwrite this method because we need to manually overwrite the buffer size
    // because it is not supposed to stay the same as the Classifier if user sets it to -1.
    transformSchema(dataset.schema, logging = true)
    val model = train(dataset)
    val bufferSize = model.getBufferSize
    copyValues(model.setParent(this)).setBufferSize(bufferSize)
  }

  override def copy(extra: ParamMap): KNNClassifier = defaultCopy(extra)
}

class KNNClassificationModel private[ml](
                                          override val uid: String,
                                          val topTree: Broadcast[Tree],
                                          val subTrees: RDD[Tree],
                                          val _numClasses: Int
                                        ) extends ProbabilisticClassificationModel[Vector, KNNClassificationModel]
with KNNModelParams with HasWeightCol with Serializable with MLWritable {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  override def numClasses: Int = _numClasses

  // TODO: This can benefit from DataSet API
  override def transform(dataset: Dataset[_]): DataFrame = {
    val getWeight: Row => Double = {
      if($(weightCol).isEmpty) {
        r => 1.0
      } else {
        // r => r.getDouble(1)
        r => r.getAs[Double]($(weightCol))
      }
    }

    val neighborRDD : RDD[(Long, Array[(Row, Double)])] = transform(dataset, topTree, subTrees)
    val merged = neighborRDD
      .map {
        case (id, labelsDists) =>
          // `labels` is a Seq[Row[class_id, ?]]
          val (labels, _) = labelsDists.unzip
          val vector = new Array[Double](numClasses)
          var i = 0
          // N.B. No distance attenuation in this implementation.
          while (i < labels.length) {
            vector(labels(i).getDouble(0).toInt) += getWeight(labels(i))
            i += 1
          }
          val rawPrediction = Vectors.dense(vector)
          lazy val probability = raw2probability(rawPrediction)
          lazy val prediction = probability2prediction(probability)

          val values = new ArrayBuffer[Any]
          if ($(rawPredictionCol).nonEmpty) {
            values.append(rawPrediction)
          }
          if ($(probabilityCol).nonEmpty) {
            values.append(probability)
          }
          if ($(predictionCol).nonEmpty) {
            values.append(prediction)
          }

          (id, values)
      }

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(merged) //make sure we don't lose any observations
        .map {
        case (i, (row, values)) => Row.fromSeq(row.toSeq ++ values.get)
      },
      transformSchema(dataset.schema)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    var transformed = schema
    if ($(rawPredictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(rawPredictionCol), new VectorUDT)
    }
    if ($(probabilityCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(probabilityCol), new VectorUDT)
    }
    if ($(predictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(predictionCol), DoubleType)
    }
    transformed
  }

  override def copy(extra: ParamMap): KNNClassificationModel = {
    val copied = new KNNClassificationModel(uid, topTree, subTrees, numClasses)
    copyValues(copied, extra).setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size

        var sum = 0.0
        while (i < size) {
          sum += dv.values(i)
          i += 1
        }

        i = 0
        while (i < size) {
          dv.values(i) /= sum
          i += 1
        }

        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in KNNClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }

  override def write: MLWriter = new KNNClassificationModelWriter(this)

}

object KNNClassificationModel extends MLReadable[KNNClassificationModel] {
  
  private case class StData(tree: Tree)
  private case class TtData(tree: Tree)

  class KNNClassificationModelWriter(instance: KNNClassificationModel) extends MLWriter {

    val logger = log4j.Logger.getLogger(getClass)

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata = ("numClasses" -> instance.numClasses) ~ ("numSubTrees" -> instance.subTrees.count)
      org.apache.spark.persistence.knn.DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
      val ttData = TtData(instance.topTree.value)
      val ttPath = new Path(path, "topTree").toString

      FsagSerialization.fsagSerializeObject(instance.topTree.value, ttPath)

      val c = instance.subTrees.count
      val stPath = new Path(path, "subTrees").toString
      val res = instance.subTrees.zipWithIndex().map { t => {
        val i = t._2
        val stPath = new Path(path, s"subTrees/$i").toString
        FsagSerialization.fsagSerializeObject(t._1, stPath)
        stPath
      }}.collect
      logger.info(s"I wrote subtrees $res.")
    }
  }

  private class KNNClassificationModelReader private[ml] extends MLReader[KNNClassificationModel] {

    val logger = log4j.Logger.getLogger(getClass)

    private val className = classOf[KNNClassificationModel].getName

    override def load(path: String): KNNClassificationModel = {
      val metadata = org.apache.spark.persistence.knn.DefaultParamsReader.loadMetadata(path, sc, className)

      implicit val format = DefaultFormats

      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numSubTrees = (metadata.metadata \ "numSubTrees").extract[Int]

      val ttPath = new Path(path, "topTree").toString
      /* val topTree = sparkSession.read.parquet(tlDataPath)
        .select("tree")
        .head()
        .getAs[Tree](0) */
      val topTree: Tree = FsagSerialization.fsagDeserializeObject(ttPath).asInstanceOf[Tree]

      var subTrees: RDD[Tree] = sc.emptyRDD
      if (numSubTrees > 0) {
        val stPath = new Path(path, "subTrees").toString
        val stPaths = FsagSerialization.fsagLs(stPath)
        /*val subTreesArr: Seq[Tree] = stPaths.map { case p: String =>
          FsagSerialization.fsagDeserializeObject(p).asInstanceOf[Tree]
        }
        subTrees = sc.parallelize(subTreesArr).persist(StorageLevel.MEMORY_AND_DISK)*/

        val fnRdd = sc.parallelize(stPaths, numSubTrees)
        subTrees = fnRdd.map( p => {
          FsagSerialization.fsagDeserializeObject(p).asInstanceOf[Tree]
        }).persist(StorageLevel.MEMORY_AND_DISK)
        if (subTrees.isEmpty) {
          logger.error(s"I tried to distribute $numSubTrees subtrees, but I got an empty RDD.")
          throw new IOException("Error deserializing subtrees.")
        }
      } else {
        logger.error(s"No subtrees indicated in metadata. Corrupt save?")
        throw new IOException("No subtrees in model metadata.")
      }

      val c = subTrees.count
      logger.info(s"I made $c subtrees.")

      val model = new KNNClassificationModel(metadata.uid, sc.broadcast(topTree), subTrees, numClasses)
      org.apache.spark.persistence.knn.DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[KNNClassificationModel] = new KNNClassificationModelReader

  override def load(path: String): KNNClassificationModel = super.load(path)
}