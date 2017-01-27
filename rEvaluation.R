evaluateModel <- function(data,results) {
    confMatrix <- table(data,results)
    err <- (confMatrix["1","0"]+confMatrix["0","1"])/sum(confMatrix)
    kappa <- vcd::Kappa(confMatrix)
    kappa <- kappa$Unweighted[1]

    names(err) <- c("Error")
    names(kappa) <- c("kappa")

    results <- list(err, kappa)
    results
}

evaluateAllTheThings <- function(groundTruth, prediction){
    f1 <- MLmetrics::F1_Score(y_pred = prediction, y_true = groundTruth)
    auc <- MLmetrics::AUC(y_pred = prediction, y_true = groundTruth)
    names(f1) <- c("f1_R")
    names(auc) <- c("AUC_R")

    evalA <- evaluateModel(groundTruth,prediction)
    index <- length(evalA)+1

    evalA[[index]] <- f1
    evalA[[index+1]] <- auc
    evalA
}
