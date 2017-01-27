import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostEstimator}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.ddahl.rscala.RClient

import scala.concurrent.duration._
import scala.language.postfixOps

case class MeasureUnit(name: String, value: Double)

object XgboostComparison extends App {

  val logger: Logger = Logger.getLogger(this.getClass)
  Logger.getLogger("org").setLevel(Level.WARN)
  val conf: SparkConf = new SparkConf()
    .setAppName("xgboostComparison")
    .setMaster("local[*]")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  val spark: SparkSession = SparkSession
    .builder()
    .config(conf)
    .getOrCreate()

  import spark.implicits._

  val mySeed = 45

  val xgbBaseParams = Map(
    "max_depth" -> 2,
    "num_rounds" -> 2,
    "eta" -> 0.01,
    "gamma" -> 0.0,
    "subsample" -> 0.7,
    "colsample_bytree" -> 0.7,
    "colsample_bylevel" -> 0.6,
    "min_child_weight" -> 1,
    "max_delta_step" -> 0,
    "seed" -> mySeed,
    "eval_metric" -> "error",
    "seed" -> mySeed,
    "scale_pos_weight" -> 1,
    "silent" -> 1,
    "lambda" -> 2.0,
    "alpha" -> 0.0,
    "boosterType" -> "gbtree",
    "useExternalMemory" -> false,
    "objective" -> "binary:logistic",
    "tracker_conf" -> TrackerConf(1 minute, "scala")
  )

  val xgbEstimator = new XGBoostEstimator(xgbBaseParams)
    .setFeaturesCol("features")
    .setLabelCol("target")

  val R = setupREvaluation()

  for (s <- 1 to 5) {
    logger.warn("%%%%%%%%%%%%%%%%%%%%% Fold: " + s + " %%%%%%%%%%%%%%%%%%%%%%%%%")

    val inputTrain = "clean_train_" + s + "_.csv"
    val inputTest = "clean_test_" + s + "_.csv"
    val fitted1Train = spark.read.
      option("header", "true")
      .option("inferSchema", true)
      .option("charset", "UTF-8")
      .option("delimiter", ";")
      .csv(inputTrain)

    val fitted1Test = spark.read.
      option("header", "true")
      .option("inferSchema", true)
      .option("charset", "UTF-8")
      .option("delimiter", ";")
      .csv(inputTest)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(fitted1Train.columns.filter(_ != "target"))
      .setOutputCol("features")

    val fittedPipeline2 = new Pipeline().
      setStages(Array(vectorAssembler, xgbEstimator))
      .fit(fitted1Train)

    val testPerformance = fittedPipeline2
      .transform(fitted1Test)
      .select("target", "probabilities", "prediction")

    calculateEvaluation(R, testPerformance, logger)
  }

  spark.stop

  def calculateEvaluation(R: RClient, testPerformance: DataFrame, logger: Logger): Seq[MeasureUnit] = {
    val (res: Array[Int], res2: Array[Int]) = testPerformance.withColumn("prediction", $"prediction".cast("Int"))
      .select("target", "prediction").as[(Int, Int)]
      .collect
      .unzip

    logger.warn("xgboost predicted number of target == 1: " + res2.sum)

    print(res2)
    print(res)

    R.y_true = res
    R.y_predicted = res2
    R.eval("evalResult <- evaluateAllTheThings(y_true, y_predicted)")
    R.eval("namesEvalRes <- names(evalResult)")

    val evalRes: (Any, String) = R.get("evalResult")
    val evalResNames: (Any, String) = R.get("namesEvalRes")

    val named = (evalResNames._1.asInstanceOf[Array[String]], evalRes._1.asInstanceOf[Array[Double]]).zipped.map {
      (m, a) => MeasureUnit(m, a)
    }
    named.foreach(println)
    named
  }

  def setupREvaluation(): RClient = {
    val R = RClient() // initialise an R interpreter
    R.eval(
      """
        |evaluateModel <- function(data,results) {
        |    confMatrix <- table(data,results)
        |    err <- (confMatrix["1","0"]+confMatrix["0","1"])/sum(confMatrix)
        |
        |    kappa <- vcd::Kappa(confMatrix)
        |    kappa <- kappa$Unweighted[1]
        |
        |    names(kappa) <- c("kappa")
        |    names(err) <- c("Error rate")
        |
        |    results <- list(err, kappa)
        |    results
        |}
        |
        |evaluateAllTheThings <- function(groundTruth, prediction){
        |    f1 <- MLmetrics::F1_Score(y_pred = prediction, y_true = groundTruth)
        |    auc <- MLmetrics::AUC(y_pred = prediction, y_true = groundTruth)
        |    names(f1) <- c("f1_R")
        |    names(auc) <- c("AUC_R")
        |
        |    evalA <- evaluateModel(groundTruth,prediction)
        |    index <- length(evalA)+1
        |    evalA[[index]] <- f1
        |    evalA[[index+1]] <- auc
        |
        |    unlist(evalA)
        |}
      """.stripMargin
    )
    R
  }
}
