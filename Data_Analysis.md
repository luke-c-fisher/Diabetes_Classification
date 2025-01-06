---
title: "Data Analysis"
author: "Luke Fisher"
date: "2025-01-06"
output: 
  html_document:
    keep_md: true
---




```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(ggplot2)
library(ISLR)
library(tibble)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(tidyr)
library(skimr)
```

## Load in data 

```r
diabetesData <- read.csv('/Users/lukefisher/Desktop/Coding/repos/Health_Analytics/Data/Diabetes_Indicators_Binary.csv')
```

## Data Wrangling

```r
diabetesData <- diabetesData %>% 
rename(Diabetes = Diabetes_binary) %>%
mutate(Diabetes = factor(Diabetes, levels = c(0, 1), labels = c("no", "yes")))

str(diabetesData)
```

```
## 'data.frame':	70692 obs. of  22 variables:
##  $ Diabetes            : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ HighBP              : num  1 1 0 1 0 0 0 0 0 0 ...
##  $ HighChol            : num  0 1 0 1 0 0 1 0 0 0 ...
##  $ CholCheck           : num  1 1 1 1 1 1 1 1 1 1 ...
##  $ BMI                 : num  26 26 26 28 29 18 26 31 32 27 ...
##  $ Smoker              : num  0 1 0 1 1 0 1 1 0 1 ...
##  $ Stroke              : num  0 1 0 0 0 0 0 0 0 0 ...
##  $ HeartDiseaseorAttack: num  0 0 0 0 0 0 0 0 0 0 ...
##  $ PhysActivity        : num  1 0 1 1 1 1 1 0 1 0 ...
##  $ Fruits              : num  0 1 1 1 1 1 1 1 1 1 ...
##  $ Veggies             : num  1 0 1 1 1 1 1 1 1 1 ...
##  $ HvyAlcoholConsump   : num  0 0 0 0 0 0 1 0 0 0 ...
##  $ AnyHealthcare       : num  1 1 1 1 1 0 1 1 1 1 ...
##  $ NoDocbcCost         : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ GenHlth             : num  3 3 1 3 2 2 1 4 3 3 ...
##  $ MentHlth            : num  5 0 0 0 0 7 0 0 0 0 ...
##  $ PhysHlth            : num  30 0 10 3 0 0 0 0 0 6 ...
##  $ DiffWalk            : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ Sex                 : num  1 1 1 1 0 0 1 1 0 1 ...
##  $ Age                 : num  4 12 13 11 8 1 13 6 3 6 ...
##  $ Education           : num  6 6 6 6 5 4 5 4 6 4 ...
##  $ Income              : num  8 8 8 8 8 7 6 3 8 4 ...
```

## Split the data into an 80/20 train vs. test split. Set the seed for replicability.

```r
set.seed(44222)

diabetesIdx = sample((nrow(diabetesData)), size = 0.8 * nrow(diabetesData))
diabetesTrn = diabetesData[diabetesIdx, ]
diabetesTst = diabetesData[-diabetesIdx, ]

head(diabetesTrn, n = 10)
```

```
##       Diabetes HighBP HighChol CholCheck BMI Smoker Stroke HeartDiseaseorAttack
## 36027      yes      0        0         1  32      0      0                    0
## 32605       no      1        0         1  29      1      0                    1
## 67519      yes      1        0         1  30      1      1                    0
## 41322      yes      1        0         1  24      1      0                    0
## 54098      yes      1        1         1  28      1      1                    1
## 34711       no      0        0         1  26      1      0                    0
## 27963       no      0        0         1  36      1      0                    0
## 12132       no      0        0         1  28      1      0                    0
## 11078       no      0        0         0  22      0      0                    0
## 38966      yes      1        0         1  44      0      0                    0
##       PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost
## 36027            1      1       1                 0             1           0
## 32605            1      0       1                 0             1           0
## 67519            1      0       1                 0             1           0
## 41322            1      1       1                 0             1           0
## 54098            1      1       1                 0             1           0
## 34711            1      1       1                 0             0           0
## 27963            1      0       1                 0             1           0
## 12132            1      1       1                 0             1           0
## 11078            1      1       1                 0             0           1
## 38966            1      1       1                 0             1           0
##       GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income
## 36027       3        0        0        0   1  11         5      7
## 32605       5        0       28        1   1   9         4      6
## 67519       5        0       30        1   1   8         6      7
## 41322       2        0        0        0   1   9         4      3
## 54098       4        0        0        0   0  13         5      6
## 34711       3        0        0        0   0   3         4      3
## 27963       3        0        2        0   1   6         5      8
## 12132       2        0        0        1   0   8         5      4
## 11078       1        0        0        0   1   6         4      3
## 38966       3        2        1        1   0   8         3      3
```

## Run a series of logistic regressions

```r
mod1 <- glm(Diabetes ~ HighBP, data = diabetesTrn, family = "binomial")
mod2 <- glm(Diabetes ~ HighBP + Smoker, data = diabetesTrn, family = "binomial")
mod3 <- glm(Diabetes ~ HighBP + Smoker + Stroke, data = diabetesTrn, family = "binomial")
mod4 <- glm(Diabetes ~ HighBP + Smoker + Stroke + BMI, data = diabetesTrn, family = "binomial")
mod5 <- glm(Diabetes ~ HighBP + Smoker + Stroke + BMI + PhysActivity + PhysHlth, data = diabetesTrn, family = "binomial")
```

## Create eight total confusion matrices, 
## four by applying your models to the training data, and four by applying your models to the test data. Use a Bayes Classifier. 
## Briefly discuss your findings. How does the error rate, sensitivity, and specificity change as the number of predictors increases?

```r
# Use lapply to predict on train and test data

modelList <- list (mod1, mod2, mod3, mod4, mod5)

trnPred <- lapply(modelList, function(mod){
    ifelse(predict(mod, newdata = diabetesTrn, type = "response") > 0.5, "yes", "no")
})

tstPred <- lapply(modelList, function(mod){
    ifelse(predict(mod, newdata = diabetesTst, type = "response") > 0.5, "yes", "no")
})

# Create tables to get the actual vs predicted values 
trnTables <- lapply(trnPred, function(pred){
    table(predicted = pred, actual = diabetesTrn$Diabetes)
})

tstTables <- lapply(tstPred, function(pred){
    table(predicted = pred, actual = diabetesTst$Diabetes)
})

# Use predictions to develop eight confusion matrices, four for the train data and four for the test data
confTrn1 <- confusionMatrix(trnTables[[1]], response = "yes")
confTrn2 <- confusionMatrix(trnTables[[2]], response = "yes")
confTrn3 <- confusionMatrix(trnTables[[3]], response = "yes")
confTrn4 <- confusionMatrix(trnTables[[4]], response = "yes")
confTrn5 <- confusionMatrix(trnTables[[5]], response = "yes")

confTst1 <- confusionMatrix(tstTables[[1]], response = "yes")
confTst2 <- confusionMatrix(tstTables[[2]], response = "yes")
confTst3 <- confusionMatrix(tstTables[[3]], response = "yes")
confTst4 <- confusionMatrix(tstTables[[4]], response = "yes")
confTst5 <- confusionMatrix(tstTables[[5]], response = "yes")
```

# Create a combined matrix of the confusion matrices 

```r
# Train
trnMatrx1 <- data.frame( 
  Model = "Train Model 1",
  Accuracy = confTrn1$overall['Accuracy'],
  Sensitivity = confTrn1$byClass['Sensitivity'],
  Specificity = confTrn1$byClass['Specificity'])

trnMatrx2 <- data.frame( 
  Model = "Train Model 2",
  Accuracy = confTrn2$overall['Accuracy'],
  Sensitivity = confTrn2$byClass['Sensitivity'],
  Specificity = confTrn2$byClass['Specificity'])
            
trnMatrx3 <- data.frame( 
  Model = "Train Model 3",
  Accuracy = confTrn3$overall['Accuracy'],
  Sensitivity = confTrn3$byClass['Sensitivity'],
  Specificity = confTrn3$byClass['Specificity'])

trnMatrx4 <- data.frame(
  Model = "Train Model 4",
  Accuracy = confTrn4$overall['Accuracy'],
  Sensitivity = confTrn4$byClass['Sensitivity'],
  Specificity = confTrn4$byClass['Specificity'])

trnMatrx5 <- data.frame(
  Model = "Train Model 5",
  Accuracy = confTrn5$overall['Accuracy'],
  Sensitivity = confTrn5$byClass['Sensitivity'],
  Specificity = confTrn5$byClass['Specificity'])


# Test
tstMatrx1 <- data.frame(
  Model = "Test Model 1",
  Accuracy = confTst1$overall['Accuracy'],
  Sensitivity = confTst1$byClass['Sensitivity'],
  Specificity = confTst1$byClass['Specificity'])

tstMatrx2 <- data.frame(
  Model = "Test Model 2",
  Accuracy = confTst2$overall['Accuracy'],
  Sensitivity = confTst2$byClass['Sensitivity'],
  Specificity = confTst2$byClass['Specificity'])
        
tstMatrx3 <- data.frame(
  Model = "Test Model 3",
  Accuracy = confTst3$overall['Accuracy'],
  Sensitivity = confTst3$byClass['Sensitivity'],
  Specificity = confTst3$byClass['Specificity'])

tstMatrx4 <- data.frame(
  Model = "Test Model 4",
  Accuracy = confTst4$overall['Accuracy'],
  Sensitivity = confTst4$byClass['Sensitivity'],
  Specificity = confTst4$byClass['Specificity'])

tstMatrx5 <- data.frame(
  Model = "Train Model 5",
  Accuracy = confTst5$overall['Accuracy'],
  Sensitivity = confTst5$byClass['Sensitivity'],
  Specificity = confTst5$byClass['Specificity'])

combinedTrnMatrx <- rbind(trnMatrx1, trnMatrx2, trnMatrx3, trnMatrx4, trnMatrx5)
combinedTstMatrx <- rbind(tstMatrx1, tstMatrx2, tstMatrx3, tstMatrx4, tstMatrx5)

print(combinedTrnMatrx)
```

```
##                   Model  Accuracy Sensitivity Specificity
## Accuracy  Train Model 1 0.6893180   0.6251859   0.7533119
## Accuracy1 Train Model 2 0.6893180   0.6251859   0.7533119
## Accuracy2 Train Model 3 0.6893180   0.6251859   0.7533119
## Accuracy3 Train Model 4 0.6983184   0.6536147   0.7429258
## Accuracy4 Train Model 5 0.7044896   0.6708561   0.7380507
```

```r
print(combinedTstMatrx)
```

```
##                   Model  Accuracy Sensitivity Specificity
## Accuracy   Test Model 1 0.6888040   0.6280282   0.7501065
## Accuracy1  Test Model 2 0.6888040   0.6280282   0.7501065
## Accuracy2  Test Model 3 0.6888040   0.6280282   0.7501065
## Accuracy3  Test Model 4 0.6979984   0.6553521   0.7410143
## Accuracy4 Train Model 5 0.7063442   0.6752113   0.7377468
```

Without adjusting the cutoff, we see accuracy and sensitivity grow while specificity falls 
as the number of predictors increase. 
This occurs as more confounding variables are accounted for with more predictors, lessening the amount of error in the regression. 
Sensitivity grows for this same reason. Specificity decreases due to the risk of overfitting. 
This occurs when the model starts to capture noise instead of an underlying pattern, reducing the model's ability to detect true negatives. 

## Use multiple cutoffs in a model including all predictors and report the results 


```r
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
metrics
```

```
##           Accuracy Sensitivity Specificity
## c = 0.10 0.5932527   0.9930388   0.1969014
## c = 0.33 0.7356249   0.9041057   0.5685915
## c = 0.50 0.7483556   0.7675806   0.7292958
## c = 0.66 0.7161044   0.5726666   0.8583099
## c = 0.90 0.5513120   0.1128001   0.9860563
```

## Visualize the data above using an ROC curve. 


```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
```

```
## 
## Attaching package: 'pROC'
```

```
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

```r
testProb = predict(lrgModel, newdata = diabetesTst, type = "response")
testRoc = roc(diabetesTst$Diabetes ~ testProb, plot = TRUE, print.auc = TRUE)
```

```
## Setting levels: control = no, case = yes
```

```
## Setting direction: controls < cases
```

![](Plots/Data_Analysis-unnamed-chunk-9-1.png)<!-- -->

```r
as.numeric(testRoc$auc)
```

```
## [1] 0.8275798
```

The following ROC curve automizes the above process by accounting for Sensitivity and Specificity 
at each cutoff. The optimal point on the ROC curve has a cutoff of 0.5, 
where the sensitivity is roughly 0.76. This follows the idea that the peak of the ROC curve
is the optimal balance between Sensitivity and Specificity. The implication
is that a 0.5 cutoff does the best job at capturing the most true positives and negatives in the confusion matrix. 
