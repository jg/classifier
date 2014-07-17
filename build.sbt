name := "classifier"

version := "1.0"

scalaVersion := "2.10.3"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "1.9.1" % "test"

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.10"

libraryDependencies += "nz.ac.waikato.cms.weka" % "LibLINEAR" % "1.0.2"

libraryDependencies += "de.bwaldvogel" % "liblinear" % "1.94"

libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.0.5"

scalaSource in Compile := baseDirectory.value / "src"

scalaSource in Test := baseDirectory.value / "test"
