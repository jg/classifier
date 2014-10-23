package msc
package classifier

import weka.core._
import weka.core.converters._
import weka.classifiers.trees._
import weka.filters._
import weka.filters.unsupervised.attribute._
import java.io._
import weka.classifiers.functions.LibLINEAR
import weka.classifiers.Evaluation
import java.util.Random
import weka.core.converters.ConverterUtils.DataSource

object Scratch {
  type TrainingSet = Instances
  type TestSet = Instances
  type DatasetCallback[A] = (TrainingSet, TestSet) => A
  val reutersDatasetPath = "datasets/reuters21578-modApte/"

  def loadData(): Instances = {
    val loader: TextDirectoryLoader = new TextDirectoryLoader()
    loader.setDirectory(new File("datasets/guardian.co.uk/"))
    val dataRaw: Instances = loader.getDataSet()
    val filter: StringToWordVector = new StringToWordVector()
    filter.setInputFormat(dataRaw)
    Filter.useFilter(dataRaw, filter)
  }

  // Reads in reuters21578-modApte dataset
  def withReutersData[A](trainFileName: String, testFileName: String)
                        (f: DatasetCallback[A]): Unit = {
    
    val train = loadReutersFile(trainFileName)
    val test = loadReutersFile(testFileName)
    f(train, test)
    // val fileList = new File(datasetPath).listFiles().filter { (file: File) =>
    //   file.getName.matches(".*.arff")
    // }

    // for (file <- fileList) {
    //    new DataSource(fileName).getDataSet()
    //   source.getDataSet()
    // }
  }

  def loadReutersFile(fileName: String): Instances = {
    val instances =
      new DataSource(reutersDatasetPath + fileName).getDataSet()
    instances.setClassIndex(0)

    instances
  }

  def randomizeData(data: Instances): Instances = {
    val rand: Random = new Random(231)   // create seeded number generator
    val randData = new Instances(data)   // create copy of original data
    randData.randomize(rand)             // randomize data with number generator
    randData
  }

  def runCV[A](data: Instances, folds: Integer)
           (f: DatasetCallback[A]): List[A] = {
    val randData = randomizeData(data)

    val lst = for ( n <- 0 until folds ) yield {
      val train: Instances = randData.trainCV(folds, n)
      val test: Instances = randData.testCV(folds, n)

      f(train, test)
    }

    lst.toList
  }

  def run() = {
    // val data = loadData()

    // val lst = runCV(data, 2){ (train, test) => {
    //                            val evaluation = new Evaluation(data)
    //                            val classifier: LibLINEAR = new LibLINEAR()
    //                            classifier.buildClassifier(train)
    //                            evaluation.evaluateModel(classifier, test)
    //                            println(evaluation.toSummaryString())
    //                          }

    // Instances trainInstances = ... instances got from somewhere
    // Instances testInstances = ... instances got from somewhere
    // Classifier scheme = ... scheme got from somewhere
    
    // Evaluation evaluation = new Evaluation(trainInstances);
    // evaluation.evaluateModel(scheme, testInstances);
    // System.out.println(evaluation.toSummaryString());
    val data = withReutersData("reutersAcqModApteTest-FullVocab.arff",
                               "reutersAcqModApteTrain-FullVocab.arff") {
      (train, test) => {
        val classifier: LibLINEAR = new LibLINEAR()
        classifier.buildClassifier(train)
        val evaluation = new Evaluation(train)
        evaluation.evaluateModel(classifier, test)
        val precision: Double = evaluation.precision(0)
        val recall: Double = evaluation.recall(0)
        val fMeasure: Double = evaluation.fMeasure(0)
        val confusionMatrix: Array[Array[Double]] = evaluation.confusionMatrix()
        println(f"Precision: $precision%2.3f")
        println(f"Recall: $recall%2.3f")
        println(f"F-Measure: $fMeasure%2.3f")

        println()
        println("Confusion Matrix: ")
        println(f"${confusionMatrix(0)(0)}%4.0f ${confusionMatrix(0)(1)}%4.0f")
        println(f"${confusionMatrix(1)(0)}%4.0f ${confusionMatrix(1)(1)}%4.0f")
      }
    }

    // val lst = runCV(data, 2){ (train, test) => {
    //                            val evaluation = new Evaluation(data)
    //                            val classifier: LibLINEAR = new LibLINEAR()
    //                            classifier.buildClassifier(train)
    //                            evaluation.evaluateModel(classifier, test)
    //                            println(evaluation.toSummaryString())
    //                          }
    // }
  }
  // System.out.println(evaluation.toSummaryString());

  // System.out.println();
  // System.out.println("=== Setup run " + (i+1) + " ===");
  // System.out.println("Classifier: " + classifier.getClass().getName() + " " + Utils.joinOptions(classifier.getOptions()));
  // System.out.println("Dataset: " + data.relationName());
  // System.out.println("Folds: " + folds);
  // System.out.println("Seed: " + seed);
  // System.out.println();
  // System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false));

}



