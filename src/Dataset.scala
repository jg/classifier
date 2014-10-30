package msc

import weka.core.Instances

object Dataset {
  type TrainingSet = Instances
  type TestSet = Instances
  type Dataset = Instances
  type DatasetCallback[A] = (TrainingSet, TestSet) => A
  type FileName = String
}
