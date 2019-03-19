name := "taxman"

version := "0.1"

scalaVersion := "2.12.4"

libraryDependencies ++= {

  Seq(
    // dynet
    "org.clulab" %% "fatdynet" % "0.2.0",

    "org.clulab" %% "processors-main" % "7.5.1",
  )

}