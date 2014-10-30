package msc

import weka.core.Instances
import java.util.Random
import msc.Dataset._

object Evaluation {
  def randomizeData(data: Instances): Instances = {
    val rand: Random = new Random(231)   // create seeded number generator
    val randData = new Instances(data)   // create copy of original data
    randData.randomize(rand)             // randomize data with number generator
    randData
  }

  def runCV[A](data: Instances, folds: Integer)
           (f: (Instances, Instances, Integer) => A): List[A] = {
    val randData = randomizeData(data)

    val lst = for ( n <- 1 until folds ) yield {
      val train: Instances = randData.trainCV(folds, n)
      val test: Instances = randData.testCV(folds, n)

      f(train, test, n)
    }

    lst.toList
  }


  def printEvaluationMeasures(evaluation: weka.classifiers.Evaluation): Unit = {
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

  def savePrecisionRecallCurveToFile(eval: weka.classifiers.Evaluation,
                                     fileName: String) = {
    import java.awt._
    import java.io._
    import java.util._
    import javax.swing._
    import weka.core._
    import weka.classifiers._
    import weka.classifiers.bayes.NaiveBayes;
    import weka.classifiers.evaluation.Evaluation;
    import weka.classifiers.evaluation.ThresholdCurve;
    import weka.gui.visualize._

    // generate curve
    val tc: ThresholdCurve = new ThresholdCurve()
    val classIndex = 0
    val curve: Instances = tc.getCurve(eval.predictions(), classIndex)
    val predictionCount = curve.numInstances()
    println(s"Generating curve from $predictionCount predictions")
 
    // plot curve
    val vmc: ThresholdVisualizePanel = new ThresholdVisualizePanel()
    val plotdata: PlotData2D = new PlotData2D(curve)
    plotdata.setPlotName(curve.relationName())
    plotdata.addInstanceNumberAttribute()
 
    // specify which points are connected
    val cp: Array[Boolean] = new Array(curve.numInstances())
    for (n <- 1 to cp.length-1) {
      cp(n) = true
    }

    plotdata.setConnectPoints(cp)

    // add plot
    vmc.addPlot(plotdata)
 
    // We want a precision-recall curve
    vmc.setXIndex(curve.attribute("Recall").index())
    vmc.setYIndex(curve.attribute("Precision").index())
 
    // Make window with plot but don't show it
    val jf: JFrame =  new JFrame()
    jf.setSize(500,400)
    jf.getContentPane().add(vmc)
    jf.pack()
 
    // Save to file specified as second argument (can use any of
    // BMPWriter, JPEGWriter, PNGWriter, PostscriptWriter for different formats)
    val jcw: JComponentWriter =
      new JPEGWriter(vmc.getPlotPanel(), new File(fileName))
    jcw.toOutput()
  }

  def saveROCCurveToFile(eval: weka.classifiers.Evaluation, fileName: String) = {
    import java.awt._
    import java.io._
    import java.util._
    import javax.swing._
    import weka.core._
    import weka.classifiers._
    import weka.classifiers.bayes.NaiveBayes
    import weka.classifiers.evaluation.Evaluation
    import weka.classifiers.evaluation.ThresholdCurve
    import weka.gui.visualize._

    val tc:ThresholdCurve = new ThresholdCurve()
    val classIndex = 0
    val result: Instances = tc.getCurve(eval.predictions(), classIndex)
    
    // plot curve
    val vmc :ThresholdVisualizePanel = new ThresholdVisualizePanel()
    vmc.setROCString("(Area under ROC = " +
                       Utils.doubleToString(ThresholdCurve.getROCArea(result), 4) + ")")
    vmc.setName(result.relationName())
    val tempd: PlotData2D = new PlotData2D(result)
    tempd.setPlotName(result.relationName())
    tempd.addInstanceNumberAttribute()

    // specify which points are connected
    val cp: Array[Boolean] = new Array(result.numInstances())
    for (n <- 1 to cp.length-1) {
      cp(n) = true
    }
    tempd.setConnectPoints(cp)
    // add plot
    vmc.addPlot(tempd)
    
    // display curve
    val plotName: String = vmc.getName()
    val jf: JFrame = new JFrame("Weka Classifier Visualize: "+plotName)
    jf.setSize(500,400)
    jf.getContentPane().add(vmc, BorderLayout.CENTER)
    jf.pack()

    // Save to file specified as second argument (can use any of
    // BMPWriter, JPEGWriter, PNGWriter, PostscriptWriter for different formats)
    val jcw: JComponentWriter =
      new JPEGWriter(vmc.getPlotPanel(), new File(fileName))
    jcw.toOutput()
  }
}
