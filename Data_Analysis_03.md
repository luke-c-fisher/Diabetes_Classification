Classifying Diabetes
================
Luke Fisher
01 April, 2025

## Introduction

Diabetes is an chronic autoimmune disease affecting millions of
Americans each year. It is best described as the body’s inability to
properly produce insulin, or produce any at all. This is a result of
either an invalid or exhausted pancreas, whose job is to secrete enough
insulin to manage blood-glucose levels. Normally, insulin is released to
enable cells to absorb the blood-glucose to use for energy. In this way,
it acts as a “key” between blood-glucose and cells.

For a diabetic, however, this “key” doesn’t occur naturally, instead
taking the form of insulin injections. As such, a diabetic uses a
glucose monitor to regulate their blood sugar–whose excess or lack
thereof has detrimental consequences. For this reason, it is important
to know whether or not someone is diabetic. In this project, I will use
classification to identify diabetes.

## Data Collection

The classification will be based on a dataset from the CDCs Behavioral
Risk Factor Surveillance System (BRFSS). The data contains 70,692
responses from the 2015 BRFSS survey, each related to risk factors like
smoking, high cholesterol, and physical activity. Furthermore, the data
contains an equal 50-50 split of respondents with and without diabetes.

The data is binary, meaning that the predictors take on a value one or
zero depending on whether the condition is true or not. For instance, if
a respondent has a smoking habit they will be marked with a 1 for the
smoking column; otherwise, they will receive a 0. There are some
exceptions to this like BMI and age, where the values are continuous.

## Methodology

The classification will be done by binary logistic regression. As such,
the response variable, Diabetes, will take on two values, “yes” or “no”,
corresponding to whether the patient has the disease. The classification
will start with a series of logistic models, each with a unique cutoff.
The function will label, respectively, “yes” and “no” for instances of
Diabetes above or below a given cutoff. Afterwards, these predicted
values will be compared with the actual values in a table and put into a
confusion matrix for evaluation. The goal is to isolate and optimize one
model for classification. As such, modifications to this model will be
taken if the it exhibits unacceptable metrics.

``` r
library(dplyr)
library(ggplot2)
library(ISLR)
library(tibble)
library(caret)
library(tidyr)
library(skimr)
library(glmnet)
library(car)
library(xgboost)
library(pROC)
library(knitr)
```

``` r
diabetesData <- read.csv('/Users/lukefisher/Desktop/Coding/repos/Health_Analytics/Data/Diabetes_Indicators_Binary.csv')
```

## Data Wrangling

``` r
diabetesData <- diabetesData %>% 
rename(Diabetes = Diabetes_binary) %>%
mutate(Diabetes = factor(Diabetes, levels = c(0, 1), labels = c("no", "yes")))
```

``` r
# Split the data into an 80/20 train vs. test split. Set the seed for replicability.
set.seed(44222)

diabetesIdx = sample((nrow(diabetesData)), size = 0.8 * nrow(diabetesData))
diabetesTrn = diabetesData[diabetesIdx, ]
diabetesTst = diabetesData[-diabetesIdx, ]

dataHead <- head(diabetesTrn, n = 10)

kable(dataHead)
```

|       | Diabetes | HighBP | HighChol | CholCheck | BMI | Smoker | Stroke | HeartDiseaseorAttack | PhysActivity | Fruits | Veggies | HvyAlcoholConsump | AnyHealthcare | NoDocbcCost | GenHlth | MentHlth | PhysHlth | DiffWalk | Sex | Age | Education | Income |
|:------|:---------|-------:|---------:|----------:|----:|-------:|-------:|---------------------:|-------------:|-------:|--------:|------------------:|--------------:|------------:|--------:|---------:|---------:|---------:|----:|----:|----------:|-------:|
| 36027 | yes      |      0 |        0 |         1 |  32 |      0 |      0 |                    0 |            1 |      1 |       1 |                 0 |             1 |           0 |       3 |        0 |        0 |        0 |   1 |  11 |         5 |      7 |
| 32605 | no       |      1 |        0 |         1 |  29 |      1 |      0 |                    1 |            1 |      0 |       1 |                 0 |             1 |           0 |       5 |        0 |       28 |        1 |   1 |   9 |         4 |      6 |
| 67519 | yes      |      1 |        0 |         1 |  30 |      1 |      1 |                    0 |            1 |      0 |       1 |                 0 |             1 |           0 |       5 |        0 |       30 |        1 |   1 |   8 |         6 |      7 |
| 41322 | yes      |      1 |        0 |         1 |  24 |      1 |      0 |                    0 |            1 |      1 |       1 |                 0 |             1 |           0 |       2 |        0 |        0 |        0 |   1 |   9 |         4 |      3 |
| 54098 | yes      |      1 |        1 |         1 |  28 |      1 |      1 |                    1 |            1 |      1 |       1 |                 0 |             1 |           0 |       4 |        0 |        0 |        0 |   0 |  13 |         5 |      6 |
| 34711 | no       |      0 |        0 |         1 |  26 |      1 |      0 |                    0 |            1 |      1 |       1 |                 0 |             0 |           0 |       3 |        0 |        0 |        0 |   0 |   3 |         4 |      3 |
| 27963 | no       |      0 |        0 |         1 |  36 |      1 |      0 |                    0 |            1 |      0 |       1 |                 0 |             1 |           0 |       3 |        0 |        2 |        0 |   1 |   6 |         5 |      8 |
| 12132 | no       |      0 |        0 |         1 |  28 |      1 |      0 |                    0 |            1 |      1 |       1 |                 0 |             1 |           0 |       2 |        0 |        0 |        1 |   0 |   8 |         5 |      4 |
| 11078 | no       |      0 |        0 |         0 |  22 |      0 |      0 |                    0 |            1 |      1 |       1 |                 0 |             0 |           1 |       1 |        0 |        0 |        0 |   1 |   6 |         4 |      3 |
| 38966 | yes      |      1 |        0 |         1 |  44 |      0 |      0 |                    0 |            1 |      1 |       1 |                 0 |             1 |           0 |       3 |        2 |        1 |        1 |   0 |   8 |         3 |      3 |

## Building classifiers

``` r
get_logistic_pred = function(mod, data, res = "y", pos = 1, neg = 0, cut = 0.5) {
  probs = predict(mod, newdata = data, type = "response")
  ifelse(probs > cut, pos, neg)
}


# Creating separate predictions based on different cutoffs

lrgModel = glm(Diabetes ~ ., data = diabetesTrn, family = "binomial")

testPred_01 = get_logistic_pred(lrgModel, diabetesTst, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.1)

testPred_02 = get_logistic_pred(lrgModel, diabetesTst, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.33)

testPred_03 = get_logistic_pred(lrgModel, diabetesTst, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.5)

testPred_04 = get_logistic_pred(lrgModel, diabetesTst, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.66)

testPred_05 = get_logistic_pred(lrgModel, diabetesTst, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.9)


# Evaluate Accuaracy, Sensitivity, and Specificity for each cutoff
testTab_01 <- table(predicted = testPred_01, actual = diabetesTst$Diabetes)
testTab_02 <- table(predicted = testPred_02, actual = diabetesTst$Diabetes)
testTab_03 <- table(predicted = testPred_03, actual = diabetesTst$Diabetes)
testTab_04 <- table(predicted = testPred_04, actual = diabetesTst$Diabetes)
testTab_05 <- table(predicted = testPred_05, actual = diabetesTst$Diabetes)


testMatrx_01 <- confusionMatrix(testTab_01, positive = "yes")
testMatrx_02 <- confusionMatrix(testTab_02, positive = "yes")
testMatrx_03 <- confusionMatrix(testTab_03, positive = "yes")
testMatrx_04 <- confusionMatrix(testTab_04, positive = "yes")
testMatrx_05 <- confusionMatrix(testTab_05, positive = "yes")


metrics <- rbind(
  c(testMatrx_01$overall["Accuracy"],
    testMatrx_01$byClass["Sensitivity"],
    testMatrx_01$byClass["Specificity"]),

  c(testMatrx_02$overall["Accuracy"],
    testMatrx_02$byClass["Sensitivity"],
    testMatrx_02$byClass["Specificity"]),

  c(testMatrx_03$overall["Accuracy"],
    testMatrx_03$byClass["Sensitivity"],
    testMatrx_03$byClass["Specificity"]),
    
  c(testMatrx_04$overall["Accuracy"],
    testMatrx_04$byClass["Sensitivity"],
    testMatrx_04$byClass["Specificity"]),

  c(testMatrx_05$overall["Accuracy"],
    testMatrx_05$byClass["Sensitivity"],
    testMatrx_05$byClass["Specificity"])
)

rownames(metrics) = c("c = 0.10", "c = 0.33", "c = 0.50", "c = 0.66", "c = 0.90")

metrics_tibble <- as_tibble(metrics, rownames = "Threshold")

kable(metrics_tibble)
```

| Threshold |  Accuracy | Sensitivity | Specificity |
|:----------|----------:|------------:|------------:|
| c = 0.10  | 0.5932527 |   0.9930388 |   0.1969014 |
| c = 0.33  | 0.7356249 |   0.9041057 |   0.5685915 |
| c = 0.50  | 0.7483556 |   0.7675806 |   0.7292958 |
| c = 0.66  | 0.7161044 |   0.5726666 |   0.8583099 |
| c = 0.90  | 0.5513120 |   0.1128001 |   0.9860563 |

The table above contains regression models with varying cutoffs. The
model with a 0.5 cutoff appears to have the most balanced trade-off
between Accuracy, Specificity, and Sensitivity, exhibiting
characteristics of a valid classifier.

## Comparing test and train errors of the logistic model.

``` r
calcErr = function(actual, predicted) {
  mean(actual != predicted)
}

trainPred_03 = get_logistic_pred(lrgModel, diabetesTrn, res = "Diabetes", 
pos = "yes", neg = "no", cut = 0.5)

# Predict on the training data
trainErr_03 = calcErr(actual = diabetesTrn$Diabetes, predicted = trainPred_03)

# Calculate test error (already done in your code)
testErr_03 = calcErr(actual = diabetesTst$Diabetes, predicted = testPred_03)

# Compare train and test errors
errorComparison = tibble::tibble(
  Type = c("Train Error", "Test Error"),
  Error = c(trainErr_03, testErr_03)
)

kable(errorComparison)
```

| Type        |     Error |
|:------------|----------:|
| Train Error | 0.2518346 |
| Test Error  | 0.2516444 |

## Significance testing

``` r
# Isolate the most significant predictors

modelSum <- summary(lrgModel)


kable(modelSum$coefficients)
```

|                      |   Estimate | Std. Error |     z value | Pr(\>\|z\|) |
|:---------------------|-----------:|-----------:|------------:|------------:|
| (Intercept)          | -6.8482299 |  0.1389907 | -49.2711376 |   0.0000000 |
| HighBP               |  0.7420328 |  0.0220466 |  33.6575227 |   0.0000000 |
| HighChol             |  0.5838587 |  0.0210585 |  27.7256017 |   0.0000000 |
| CholCheck            |  1.3477969 |  0.0912931 |  14.7633983 |   0.0000000 |
| BMI                  |  0.0754391 |  0.0017637 |  42.7740408 |   0.0000000 |
| Smoker               |  0.0038118 |  0.0210863 |   0.1807712 |   0.8565472 |
| Stroke               |  0.1933263 |  0.0458602 |   4.2155562 |   0.0000249 |
| HeartDiseaseorAttack |  0.2513817 |  0.0316545 |   7.9414212 |   0.0000000 |
| PhysActivity         | -0.0243045 |  0.0238038 |  -1.0210322 |   0.3072392 |
| Fruits               | -0.0577295 |  0.0218803 |  -2.6384265 |   0.0083292 |
| Veggies              | -0.0445863 |  0.0260545 |  -1.7112709 |   0.0870311 |
| HvyAlcoholConsump    | -0.7332077 |  0.0539896 | -13.5805303 |   0.0000000 |
| AnyHealthcare        |  0.0381443 |  0.0527735 |   0.7227938 |   0.4698066 |
| NoDocbcCost          |  0.0304985 |  0.0380442 |   0.8016583 |   0.4227507 |
| GenHlth              |  0.5796055 |  0.0127947 |  45.3004758 |   0.0000000 |
| MentHlth             | -0.0040392 |  0.0014334 |  -2.8179099 |   0.0048337 |
| PhysHlth             | -0.0083631 |  0.0013315 |  -6.2809966 |   0.0000000 |
| DiffWalk             |  0.1250893 |  0.0289458 |   4.3215062 |   0.0000155 |
| Sex                  |  0.2632854 |  0.0214026 |  12.3015753 |   0.0000000 |
| Age                  |  0.1506930 |  0.0043579 |  34.5795388 |   0.0000000 |
| Education            | -0.0293567 |  0.0114374 |  -2.5667216 |   0.0102665 |
| Income               | -0.0586832 |  0.0058000 | -10.1178662 |   0.0000000 |

We can deduce `BMI`, `GenHlth`, `Age`, `HighBP`, and `HighChol` as the
most significant predictors in the initial model. As such, we will add
these predictors to the initial model, `lrgModel`, along with more
complexity.

## Data prep

``` r
# Convert training and test data to matrix format

trainMatrx = model.matrix(Diabetes ~ . + I(BMI^2) + I(GenHlth^2) + Age + HighBP + HighChol +
                          BMI:Age + HighBP:GenHlth, data = diabetesTrn)

testMatrx = model.matrix(Diabetes ~ . + I(BMI^2) + I(GenHlth^2) + Age + HighBP + HighChol +
                         BMI:Age + HighBP:GenHlth, data = diabetesTst)
```

## Creating an Boost Model

``` r
trainLabel = as.numeric(diabetesTrn$Diabetes == "yes")
testLabel = as.numeric(diabetesTst$Diabetes == "yes")

parameters <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 6,
  eta = 0.1,
  nthread = 2
)

boostMod <- xgboost(
  data = trainMatrx,
  label = trainLabel,
  params = parameters,
  nrounds = 100,
  verbose = 0
)

# Predict on the test data
tstPredictions <- predict(boostMod, testMatrx)

# Apply 0.5 cutoff
tstPredLabels <- ifelse(tstPredictions > 0.5, "yes", "no")
```

## Evaluating error from boost model

``` r
# Create predictions on the train data 
trnPredictions <- predict(boostMod, trainMatrx)

# Apply 0.5 cutoff
trnPredLabels <- ifelse(trnPredictions > 0.5, "yes", "no")

trnBoostErr = calcErr(actual = diabetesTrn$Diabetes, predicted = trnPredLabels)
tstBoostErr = calcErr(actual = diabetesTst$Diabetes, predicted = tstPredLabels)

errorComparison2 = tibble::tibble(
  Type = c("Train Error", "Test Error"),
  Error = c(trnBoostErr, tstBoostErr)
)

kable(errorComparison2)
```

| Type        |     Error |
|:------------|----------:|
| Train Error | 0.2305802 |
| Test Error  | 0.2467643 |

## Model comparison

``` r
# Create confusion matrix
boostTab <- table(Predicted = tstPredLabels, Actual = diabetesTst$Diabetes)
boostMatrx <- confusionMatrix(boostTab, positive = "yes")

boostMetrics <- rbind(
  c(testMatrx_03$overall["Accuracy"],
    testMatrx_03$byClass["Sensitivity"],
    testMatrx_03$byClass["Specificity"]),
  
  c(boostMatrx$overall["Accuracy"],
    boostMatrx$byClass["Sensitivity"],
    boostMatrx$byClass["Specificity"]))

rownames(boostMetrics) <- c("Logistic", "Boost")

metric_comparison <- as_tibble(boostMetrics, rownames = "Model")

kable(metric_comparison)
```

| Model    |  Accuracy | Sensitivity | Specificity |
|:---------|----------:|------------:|------------:|
| Logistic | 0.7483556 |   0.7675806 |   0.7292958 |
| Boost    | 0.7532357 |   0.7931524 |   0.7136620 |

## Evaluate

Two models were used to classify Diabetes, a logistic and xgboost model.
For the logistic method, models with multiple cutoffs were used to
identify the most accurate one, with the 0.5 cutoff yielding the best
results. The model exhibited the most balanced trade off between
Accuracy, Sensitivity, and Specificity, with the values, respectively,
of 0.74, 0.76, 0.72. This indicated that the model was able to identify
instances of diabetes with 74 percent Accuracy, with true positives and
negatives sitting at 76 and 72 percent, respectively. To ensure that
these metrics were not the result of under or over-fitting, the test and
train error were compared. With both values sitting around 0.25, there
was little reason to suspect a poor fit model, as such a case would
involve a large gap between the errors.

A gradient boost model using xgboost was used. In this model, “weak
learners”, or stumps from a decision tree, are aggregated in an ensemble
model. The residuals from this model are then scaled by a learning rate
and fitted to a new model. This ensures error is reduced without
over-fitting. The effects of the xgboost model are evident in the RMSE
values, with the train and test, respectively, sitting at 0.39 and 0.40,
down from 0.50 in the logistic model. Additionally, the errors are down
from 0.25, with the train and test sitting at 0.23 and 0.24,
respectively. This improvement in accuracy is reaffirmed by the
performance metrics, with the boost model leading in accuracy and
sensitivity.

## Conclusion

Both models above were able to predict Diabetes with acceptable
accuracy. However, the logistic model contained RMSE values

The boost model was able to do this without over-fitting, as the train
and test RMSE’s were close together. This improvement in accuracy was
partially due to feature selection, where the most significant
predictors in the logistic model were isolated for use in the boost
model.

The resulting accuracy was an improvement from the logistic model, as
was sensitivity. The caveat was that accuracy came at the expense of
specificity, or the model’s ability to capture true negatives. In the
context of detecting Diabetes, however, the consequences of a false
positive (specificity) carry less weight than a false negative
(sensitivity), in which case a diabetic would be incorrectly labeled as
non-diabetic. As such, specificity is less of a concern, as the
importance lies in the model’s ability to detect Diabetics, not
non-diabetics.

With an improvement in accuracy, our boost model is able to predict the
instance of Diabetes with 75 percent accuracy. With
