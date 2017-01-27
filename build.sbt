import java.io.File

name := "XgboostComparison"
organization := "myOrg"

scalaVersion := "2.11.8"

scalacOptions ++= Seq(
  "-target:jvm-1.8",
  "-encoding", "UTF-8",
  "-unchecked",
  "-deprecation",
  "-Xfuture",
  "-Xlint:missing-interpolator",
  "-Yno-adapted-args",
  "-Ywarn-dead-code",
  "-Ywarn-numeric-widen",
  "-Ywarn-value-discard",
  "-Ywarn-dead-code",
  "-Ywarn-unused",
  "-feature"
)
lazy val spark = "2.1.0"
lazy val xgboost = "0.7"

resolvers += Resolver.jcenterRepo
lazy val os = System.getProperty("os.name").toLowerCase.substring(0,3)
if (os.startsWith("win")) {
  resolvers += "Local Maven Repository" at "file:///" + Path.userHome + File.separator + ".m2" + File.separator + "repository"
} else {
  resolvers += "Local Maven Repository" at "file://" + Path.userHome + File.separator + ".m2" + File.separator + "repository"
}

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % spark % "provided",
  "org.apache.spark" %% "spark-sql" % spark % "provided",
  "org.apache.spark" %% "spark-mllib" % spark % "provided",
"ml.dmlc" % "xgboost4j-spark" % xgboost,
  "org.ddahl" %% "rscala" % "1.0.14"
)

run in Compile <<= Defaults.runTask(fullClasspath in Compile, mainClass in(Compile, run), runner in(Compile, run))


mainClass := Some("XgboostComparison")
