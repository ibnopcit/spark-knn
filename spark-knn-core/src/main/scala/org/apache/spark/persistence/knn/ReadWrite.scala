/**
 * Oct 2019 Adapted from https://raw.githubusercontent.com/raufer/custom-spark-models/master/src/main/scala/com/custom/spark/persistence/ReadWrite.scala
 */

package org.apache.spark.persistence.knn

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission
import java.nio.file.attribute.PosixFilePermissions
import java.io._

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.mllib.knn.KNNUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path, LocatedFileStatus, RemoteIterator}
import org.apache.hadoop.fs.permission.FsPermission
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, _}

import scala.collection.mutable.ArrayBuffer


class DefaultParamsWriter(instance: Params) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
  }
}

object DefaultParamsWriter {

  /**
    * Saves metadata + Params to: path + "/metadata"
    *  - class
    *  - timestamp
    *  - sparkVersion
    *  - uid
    *  - paramMap
    *  - (optionally, extra metadata)
    *
    * @param extraMetadata  Extra metadata to be saved at same level as uid, paramMap, etc.
    * @param paramMap  If given, this is saved in the "paramMap" field.
    *                  Otherwise, all [[org.apache.spark.ml.param.Param]]s are encoded using
    *                  [[org.apache.spark.ml.param.Param.jsonEncode()]].
    */
  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, sc, extraMetadata, paramMap)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
    * Helper for [[saveMetadata()]] which extracts the JSON to save.
    * This is useful for ensemble models which need to save metadata for many sub-models.
    *
    * @see [[saveMetadata()]] for details on what this includes.
    */
  def getMetadataToSave(
                         instance: Params,
                         sc: SparkContext,
                         extraMetadata: Option[JObject] = None,
                         paramMap: Option[JValue] = None): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }
}


class DefaultParamsReader[T] extends MLReader[T] {

  override def load(path: String): T = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val cls = KNNUtils.classForName(metadata.className)
    val instance =
      cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
    DefaultParamsReader.getAndSetParams(instance, metadata)
    instance.asInstanceOf[T]
  }
}

object DefaultParamsReader {

  /**
    * All info from metadata file.
    *
    * @param params  paramMap, as a `JValue`
    * @param metadata  All metadata, including the other fields
    * @param metadataJson  Full metadata file String (for debugging)
    */
  case class Metadata(
                       className: String,
                       uid: String,
                       timestamp: Long,
                       sparkVersion: String,
                       params: JValue,
                       metadata: JValue,
                       metadataJson: String) {

    /**
      * Get the JSON value of the [[org.apache.spark.ml.param.Param]] of the given name.
      * This can be useful for getting a Param value before an instance of `Params`
      * is available.
      */
    def getParamValue(paramName: String): JValue = {
      implicit val format = DefaultFormats
      params match {
        case JObject(pairs) =>
          val values = pairs.filter { case (pName, jsonValue) =>
            pName == paramName
          }.map(_._2)
          assert(values.length == 1, s"Expected one instance of Param '$paramName' but found" +
            s" ${values.length} in JSON Params: " + pairs.map(_.toString).mkString(", "))
          values.head
        case _ =>
          throw new IllegalArgumentException(
            s"Cannot recognize JSON metadata: $metadataJson.")
      }
    }
  }

  /**
    * Load metadata saved using [[DefaultParamsWriter.saveMetadata()]]
    *
    * @param expectedClassName  If non empty, this is checked against the loaded metadata.
    * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
    */
  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr, expectedClassName)
  }

  /**
    * Parse metadata JSON string produced by [[DefaultParamsWriter.getMetadataToSave()]].
    * This is a helper function for [[loadMetadata()]].
    *
    * @param metadataStr  JSON string of metadata
    * @param expectedClassName  If non empty, this is checked against the loaded metadata.
    * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
    */
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]
    val params = metadata \ "paramMap"
    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }

    Metadata(className, uid, timestamp, sparkVersion, params, metadata, metadataStr)
  }

  /**
    * Extract Params from metadata, and set them in the instance.
    * This works if all Params (except params included by `skipParams` list) implement
    * [[org.apache.spark.ml.param.Param.jsonDecode()]].
    *
    * @param skipParams The params included in `skipParams` won't be set. This is useful if some
    *                   params don't implement [[org.apache.spark.ml.param.Param.jsonDecode()]]
    *                   and need special handling.
    * TODO: Move to [[Metadata]] method
    */
  def getAndSetParams(
                       instance: Params,
                       metadata: Metadata,
                       skipParams: Option[List[String]] = None): Unit = {
    implicit val format = DefaultFormats
    metadata.params match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          if (skipParams == None || !skipParams.get.contains(paramName)) {
            val param = instance.getParam(paramName)
            val value = param.jsonDecode(compact(render(jsonValue)))
            instance.set(param, value)
          }
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
    }
  }

  /**
    * Load a `Params` instance from the given path, and return it.
    * This assumes the instance implements [[org.apache.spark.ml.util.MLReadable]].
    */
  def loadParamsInstance[T](path: String, sc: SparkContext): T = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val cls = KNNUtils.classForName(metadata.className)
    cls.getMethod("read").invoke(null).asInstanceOf[MLReader[T]].load(path)
  }
}

/**
 * Filesystem-agnostic object serialization/deserialization.
 */
object FsagSerialization {
  @throws(classOf[IOException])
  def fsagSerializeObject(o: Object, fn: String): Boolean = {
      val perms: java.util.Set[PosixFilePermission] = PosixFilePermissions.fromString("rwxrwxr--")
      val attr: FileAttribute[java.util.Set[PosixFilePermission]] = PosixFilePermissions.asFileAttribute(perms)
      fsagSerializeObject(o, fn, attr)
  }

  @throws(classOf[IOException])
  def fsagSerializeObject(o: Object, fn: String, attr: FileAttribute[java.util.Set[PosixFilePermission]]): Boolean = {
      var oo: ObjectOutputStream = null
      if (fn.startsWith("hdfs:/")) {
          val fs: FileSystem = HdfsFileSystem.getHdfsFileSystem()
          val p: Path = new Path(fn)
          val os: OutputStream = fs.create(p)
          oo = new ObjectOutputStream(os)
      } else {
          Files.createDirectories(Paths.get(FilenameUtils.getFullPath(fn)), attr)
          val f: FileOutputStream = new FileOutputStream(fn)
          oo = new ObjectOutputStream(f)
      }
      oo.writeUnshared(o)
      oo.flush()
      oo.reset()
      oo.close()
      true
  }

  @throws(classOf[IOException])
  @throws(classOf[ClassNotFoundException])
  def fsagDeserializeObject(fn: String): Object = {
    var ois: ObjectInputStream = null
    if (fn.startsWith("hdfs:/")) {
          val fs: FileSystem = HdfsFileSystem.getHdfsFileSystem()
          val p: Path = new Path(fn)
          ois = new ObjectInputStream(fs.open(p))
      } else {
          val fis: FileInputStream = new FileInputStream(fn)
          ois = new ObjectInputStream(fis)
      }
      ois.readObject()
  }

  @throws(classOf[IOException])
  def fsagLs(fn: String): Seq[String] = {
    var rval: Seq[String] = null
    if (fn.startsWith("hdfs:/")) {
        val fs: FileSystem = HdfsFileSystem.getHdfsFileSystem()
        val path: Path = new Path(fn)
        val ritr: RemoteIterator[LocatedFileStatus] = fs.listFiles(path, true)
        val ab = ArrayBuffer[String]()
        while (ritr.hasNext()) {
          val p: Path = ritr.next().getPath()
          val entry: String = s"$fn/${p.getName()}"
          ab.append(entry)
        }
        rval = ab.toSeq
    } else {
        val file: File = new File(fn)
        if (!file.exists() || !file.isDirectory()) {
            throw new FileNotFoundException(s"Directory '$fn' not found.")
        }
        if (file.exists() && file.isDirectory()) {
            rval = file.listFiles().map { case f: File =>
                f.getAbsolutePath()
            }
        }
    }
    rval
  }
}

object HdfsFileSystem {
  var fs: FileSystem = null

  @throws(classOf[IOException])
  def getHdfsFileSystem(): FileSystem = {
      if (fs == null) {
          val conf: Configuration = new Configuration(true)
          if (null != System.getProperty("oozie.action.conf.xml")) {
              conf.addResource(new Path("file:///", System.getProperty("oozie.action.conf.xml")));
          }
          fs = FileSystem.get(conf)
      }
      fs
  }
}