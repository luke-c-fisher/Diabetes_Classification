Data Analysis
================
Luke Fisher
12 March, 2025

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
model for classifying diabetes. As such, modifications to this model
will be taken if the it exhibits unacceptable metrics, such as a high
test error.

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

    ## Loading required package: lattice

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 4.1-8

    ## Warning: package 'car' was built under R version 4.3.3

    ## Loading required package: carData

    ## 
    ## Attaching package: 'car'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     recode

    ## Warning: package 'xgboost' was built under R version 4.3.3

    ## 
    ## Attaching package: 'xgboost'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

## Load in data

## Data Wrangling

    ## 'data.frame':    70692 obs. of  22 variables:
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

## Cross validation

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

    ## 
    ## 
    ## |Threshold |  Accuracy| Sensitivity| Specificity|
    ## |:---------|---------:|-----------:|-----------:|
    ## |c = 0.10  | 0.5426834|   0.9973008|   0.0919718|
    ## |c = 0.33  | 0.7131339|   0.9053843|   0.5225352|
    ## |c = 0.50  | 0.7339274|   0.7597670|   0.7083099|
    ## |c = 0.66  | 0.6940378|   0.5367240|   0.8500000|
    ## |c = 0.90  | 0.5407030|   0.0912061|   0.9863380|

The table above contains regression models with varying cutoffs. The
model with a 0.5 cutoff appears to have the most balanced trade-off
between Accuracy, Specificity, and Sensitivity, exhibiting
characteristics of a valid classifier.

## Test errors

    ## [1] 0.2660726

## Comparing test and train data to test over-fitting.

| Type        |     Error |
|:------------|----------:|
| Train Error | 0.2698707 |
| Test Error  | 0.2660726 |

Since the train and test errors are closely aligned, there is an
indication of an under-fit model. This implies that the model is too
simple and cannot capture underlying patterns in the data.

## Significance testing

|                      |   Estimate | Std. Error |    z value | Pr(\>\|z\|) |
|:---------------------|-----------:|-----------:|-----------:|------------:|
| HighBP               |  0.8635216 |  0.0215449 |  40.080106 |   0.0000000 |
| HighChol             |  0.6121934 |  0.0206149 |  29.696658 |   0.0000000 |
| CholCheck            | -0.7945084 |  0.0567606 | -13.997523 |   0.0000000 |
| BMI                  |  0.0351258 |  0.0014528 |  24.178578 |   0.0000000 |
| Smoker               | -0.1218465 |  0.0204185 |  -5.967451 |   0.0000000 |
| Stroke               |  0.1803990 |  0.0454278 |   3.971109 |   0.0000715 |
| HeartDiseaseorAttack |  0.3675319 |  0.0312398 |  11.764876 |   0.0000000 |
| PhysActivity         | -0.1984984 |  0.0228934 |  -8.670558 |   0.0000000 |
| Fruits               | -0.1150341 |  0.0211927 |  -5.428006 |   0.0000001 |
| Veggies              | -0.1700174 |  0.0253268 |  -6.712942 |   0.0000000 |
| HvyAlcoholConsump    | -0.8235246 |  0.0525319 | -15.676647 |   0.0000000 |
| AnyHealthcare        | -0.7324068 |  0.0491408 | -14.904263 |   0.0000000 |
| NoDocbcCost          | -0.3183324 |  0.0361635 |  -8.802579 |   0.0000000 |
| GenHlth              |  0.3766305 |  0.0117935 |  31.935294 |   0.0000000 |
| MentHlth             | -0.0084877 |  0.0013928 |  -6.093898 |   0.0000000 |
| PhysHlth             | -0.0026982 |  0.0012981 |  -2.078504 |   0.0376630 |
| DiffWalk             |  0.2295780 |  0.0283703 |   8.092183 |   0.0000000 |
| Sex                  |  0.1803907 |  0.0207010 |   8.714129 |   0.0000000 |
| Age                  |  0.0751976 |  0.0039056 |  19.253925 |   0.0000000 |
| Education            | -0.2553002 |  0.0103695 | -24.620400 |   0.0000000 |
| Income               | -0.0860089 |  0.0055743 | -15.429574 |   0.0000000 |

We can deduce `BMI`, `GenHlth`, `Age`, `HighBP`, and `HighChol` as the
most significant predictors in the initial model. As such, we will add
these predictors to the initial model, `lrgModel`, along with more
complexity.

## Data prep

## Using a boosting model to reduce underfitting in the model

| Model    |  Accuracy | Sensitivity | Specificity |
|:---------|----------:|------------:|------------:|
| Standard | 0.7339274 |   0.7597670 |   0.7083099 |
| Boost    | 0.7524577 |   0.7948572 |   0.7104225 |

As we can see from the comparison, the boost model delivers more
accurate results as a classifier compared to the standard model from
before.

## Evaluating error from boost model

| Type        |     Error |
|:------------|----------:|
| Train Error | 0.2220041 |
| Test Error  | 0.2475423 |

The plan above is to isolate the most significant predictors in the
initial model by measuring their p-values. The predictors with the
lowest p-values (i.e., p-value \<0.05) are added to matrices for the
boosting model. This ensures that the most significant predictors are
used, and the test error in the boosting model is lowered from its
initial value.

## Evaluate

The above model exhibits different levels of Accuracy, Sensitivity, and
Specificity at different cutoffs. This implies a change the amount of
positive and negative cases captured, (i.e., 1 for positive, 0 for
negative) meaning that the values for Accuracy, Specificity, and
Sensitivity are a direct reflection of however many positive and
negative cases there are. For example, it is no surprise that the first
model captures 99 percent of true positives under a 0.10 cutoff. It
practically only captures positive cases. The inverse is true for the
last model.

With that said, the model with the most balanced trade-off between
Accuracy, Sensitivity, and Specificity is the model with a 0.5 cutoff.
It differs from the other models in the sense that it doesn’t skew
toward one metric, making for a unbiased classifier. Furthermore, the
ROC curve hugs the top-left around the 0.50 mark, where the model
exhibits its highest true positive and negative rates. The overall
performance of the model is 0.82, meaning that it has a solid ability to
discriminate between diabetics and non-diabetics.

## Conclusion
