package msc

import weka.core.{Instance, Instances}
import weka.core.converters.TextDirectoryLoader
import java.io.File
import weka.filters.Filter
import weka.filters.unsupervised.attribute.StringToWordVector
import weka.core.converters.ConverterUtils.DataSource

import Dataset._

object Reuters {
  val reutersDatasetPath = "datasets/reuters21578-modApte/"

  // Reads in reuters21578-modApte dataset
  def withData[A](trainFileName: String, testFileName: String)
                        (f: DatasetCallback[A]): Unit = {
    
    val train = loadReutersFile(trainFileName)
    val test = loadReutersFile(testFileName)
    f(train, test)
  }

  def loadReutersFile(fileName: FileName): Instances = {
    val instances =
      new DataSource(reutersDatasetPath + fileName).getDataSet()
    instances.setClassIndex(0)

    instances
  }

  def reutersCategories: List[String] = {
    val trainFiles =
      reutersFiles.filter(_.getName.endsWith("ModApteTrain-FullVocab.arff"))
    trainFiles.map { (file: File) =>
      file.getName
        .stripPrefix("reuters")
        .stripSuffix("ModApteTrain-FullVocab.arff")
    }.toList
  }

  def testSetForCategory(category: String): TestSet = {
    val testFiles =
      reutersFiles.filter(_.getName.endsWith("ModApteTest-FullVocab.arff"))
    val file = testFiles.find(_.getName.contains(category)).get
    loadReutersFile(file.getName)
  }

  def trainingSetForCategory(category: String): TrainingSet = {
    val trainFiles =
      reutersFiles.filter(_.getName.endsWith("ModApteTrain-FullVocab.arff"))
    val file = trainFiles.find(_.getName.contains(category)).get
    loadReutersFile(file.getName)
  }

  def trainingSets: List[TrainingSet] =
    reutersCategories.map { trainingSetForCategory(_) }


  def dataSetForCategory(category: String): Dataset = {
    val train = trainingSetForCategory(category)
    val test = testSetForCategory(category)

    concatInstances(train, test)
  }

  def reutersFiles: Array[File] =
    new File("datasets/reuters21578-modApte").listFiles()

  def concatInstances(set1: Instances, set2: Instances): Instances = {
    import scala.collection.JavaConverters._
    import collection.JavaConversions.enumerationAsScalaIterator

    val set3: Instances = new Instances(set1)
    val it =
      set2.enumerateInstances().asInstanceOf[java.util.Enumeration[Instance]].asScala
    for (instance <- it) {
      set3.add(instance)
    }
    set3
  }

  def loadData(): Instances = {
    val loader: TextDirectoryLoader = new TextDirectoryLoader()
    loader.setDirectory(new File("datasets/guardian.co.uk/"))
    val dataRaw: Instances = loader.getDataSet()
    val filter: StringToWordVector = new StringToWordVector()
    filter.setInputFormat(dataRaw)
    Filter.useFilter(dataRaw, filter)
  }
}
