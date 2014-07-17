import weka.core._
import weka.core.converters._
import weka.classifiers.trees._
import weka.filters._
import weka.filters.unsupervised.attribute._
import java.io._
import weka.classifiers.functions.LibLINEAR
import weka.classifiers.Evaluation
import java.util.Random

object Scratch {
  def loadData(): Instances = {
    val loader: TextDirectoryLoader = new TextDirectoryLoader()
    loader.setDirectory(new File("datasets/guardian.co.uk/"))
    val dataRaw: Instances = loader.getDataSet()
    val filter: StringToWordVector = new StringToWordVector()
    filter.setInputFormat(dataRaw)
    Filter.useFilter(dataRaw, filter)
  }

  def randomizeData(data: Instances): Instances = {
    val rand: Random = new Random(231)   // create seeded number generator
    val randData = new Instances(data)   // create copy of original data
    randData.randomize(rand)             // randomize data with number generator
    randData
  }

  def runCV[A](data: Instances, folds: Integer)
           (f: (Instances, Instances) => A): List[A] = {
    val randData = randomizeData(data)

    val lst = for ( n <- 0 until folds ) yield {
      val train: Instances = randData.trainCV(folds, n)
      val test: Instances = randData.testCV(folds, n)

      f(train, test)
    }

    lst.toList
  }

  def run() = {
    val data = loadData()

    val lst = runCV(data, 2){ (train, test) => {
                               val evaluation = new Evaluation(data)
                               val classifier: LibLINEAR = new LibLINEAR()
                               classifier.buildClassifier(train)
                               evaluation.evaluateModel(classifier, test)
                               println(evaluation.toSummaryString())
                             }
    }
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



