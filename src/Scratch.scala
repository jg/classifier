package msc

import weka.core._
import weka.classifiers.trees._
import weka.filters._
import weka.filters.unsupervised.attribute._
import java.util.Random
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.meta.ThresholdSelector

import msc.Dataset._
import LibLinear.ThresholdOption

object Scratch {
  /*
  def processCategory(category: String) = {
    val train = Reuters.trainingSetForCategory(category)
    val test = Reuters.testSetForCategory(category)

    val c = 1
    println(s"Category: $category")
    val data = train
    val classifier = LibLinear.forTrainingSet(data, c)
    val evaluation = new weka.classifiers.Evaluation(data)
    evaluation.crossValidateModel(classifier, data, 10, new Random(1));

    println("Generating ROC Curve...")
    Evaluation.
      saveROCCurveToFile(evaluation, s"results/precision_vs_recall/$category.jpg")

    Evaluation.printEvaluationMeasures(evaluation)
  }
  */

  def processCategory(category: String) = {
    val data = Reuters.dataSetForCategory(category)
    Evaluation.runCV(data, 10) { (train, test, fold) =>
      {
        // build classifier
        val threshold = 1.0/fold
        val opts = Map("classificationThreshold" -> ThresholdOption(threshold))
        val classifier = LibLinear.forTrainingSet(train, opts)
        val evaluation = new weka.classifiers.Evaluation(test)

        evaluation.evaluateModel(classifier, test)

        // output info about model
        val precision = evaluation.precision(0)
        val recall = evaluation.recall(0)
        println(s"threshold = $threshold; precision, recall = $precision, $recall")
      }
    }
  }

  def run() = {
    val lst1 = List("Reserves", "Potato", "Ship", "Soy-oil")
    // val lst2 = List("Cpi", "Reserves")
    // for (category <- lst1) {
    //  processCategory(category)
    // }
  }

}



