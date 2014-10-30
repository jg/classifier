package msc

import weka.classifiers.functions.LibLINEAR
import weka.core.Utils.splitOptions
import weka.core.OptionHandler
import Dataset._
import scala.collection.immutable.Map
import weka.classifiers.meta.ThresholdSelector

object LibLinear {
  type ClassifierOptions = Map[String, ClassifierOption]

  abstract class ClassifierOption

  case class ThresholdOption(value: Double) extends ClassifierOption

  case class CliOption(switch: String, value: Option[Double]) extends ClassifierOption {
    override def toString = switch + " " + value.toString
  }

  def forTrainingSet(trainingSet: TrainingSet,
                     opts: ClassifierOptions): LibLINEAR = {
    val classifier =
      setClassifierOptions(new LibLINEAR(), defaultOptions ++ opts)

    classifier.buildClassifier(trainingSet)

    classifier
  }

  val defaultOptions: ClassifierOptions = Map(
    "solver"                  -> CliOption("-S", Some(0)),
    "cost"                    -> CliOption("-C", Some(1)),
    "normalization"           -> CliOption("-Z", None),
    "tolerance"               -> CliOption("-E", Some(0.01)),
    "bias"                    -> CliOption("-B", Some(1.0)),
    "probabilityEstimation"   -> CliOption("-P", None)
  )

  private def setCliOptions(classifier: LibLINEAR,
                            opts: ClassifierOptions): LibLINEAR = {
    val cliOptions = opts.filter { case (key, value) =>
      value match {
        case CliOption(_, _) => true
        case _ => false
      }
    }

    val options = cliOptions.map(_.toString).mkString(" ")

    classifier.setOptions(splitOptions(options))

    classifier
  }

  def setThreshold(classifier: LibLINEAR,
                   opts: ClassifierOptions): LibLINEAR = {
    opts.get("classificationThreshold") match {
      case Some(ThresholdOption(threshold)) =>
        val selector = new ThresholdSelector()
        selector.setClassifier(classifier)
        selector.setManualThresholdValue(threshold)
      case _ =>
    }
    classifier
  }

  private def setClassifierOptions(classifier: LibLINEAR,
                                   opts: ClassifierOptions): LibLINEAR = {
    setThreshold(setCliOptions(classifier, opts), opts)
  }
}
