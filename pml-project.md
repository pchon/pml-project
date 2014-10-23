---
Title: "PML project write-up"
---


```
## Warning: package 'caret' was built under R version 3.1.1
## Warning: package 'ggplot2' was built under R version 3.1.1
## Warning: package 'randomForest' was built under R version 3.1.1
```

## Read data & preprocessing
We load the dat into memory and remove NAs and blank cells (columns). We preprocess both training and testing data in order to have correct output. Also manually remove columns that do not contribute to the classification process.         


```r
#load data
raw <-read.csv("pml-training.csv",na.strings=c("NA",""))
rawval <-read.csv("pml-testing.csv",na.strings=c("NA",""))

#delete columns with NAs
train <-raw[, !apply( raw , 2 , function(x){any(is.na(x))})]
valid <-rawval[, !apply(rawval,2,function(x){any(is.na(x))})]

remIndex <-grep("timestamp|X|user_name|new_window",names(train))
train <-train[-remIndex]
valid <-valid[-remIndex]

dim(train)
```

```
## [1] 19622    54
```

```r
dim(valid)
```

```
## [1] 20 54
```

## Data partition and model tuning
Let's partition the train data into training and testing datasets with ratio 70% and 30% respectively.        


```r
set.seed(32343)
inTrain <-createDataPartition(y=train$classe,p=0.7,list=FALSE)
training <-train[inTrain,]
testing <-train[-inTrain,]
```

We are going to create 3 different models and validate them   against the 20 testing use cases. 

First, some model tuning. Set up a 10-fold cross-validation.


```r
fitControl <-trainControl(method="cv",number=10)
```

## Model training

Now let's train and build our three different models


```r
#logistic regression with boosting model
set.seed(32343)
model.logregboot <- train(classe~.,method="LogitBoost",data=training, trControl=fitControl)
```

```
## Loading required package: caTools
```

```
## Warning: package 'caTools' was built under R version 3.1.1
```

```r
pr.logregboot <-predict(model.logregboot, newdata=testing)
cm.logreg <- confusionMatrix(pr.logregboot,testing$classe)

#SVM 
set.seed(32343)
model.svm <-suppressWarnings(train(classe~.,data=training, method="svmRadialCost", trControl=fitControl))
```

```
## Loading required package: kernlab
```

```
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel 
## Using automatic sigma estimation (sigest) for RBF or laplace kernel
```

```r
pr.svm <-predict(model.svm, newdata=testing)
cm.cvm <-confusionMatrix(pr.svm, testing$classe)


#tree
set.seed(32343)
model.tree <-train(classe~.,method="rpart", data=training, trControl=fitControl)
```

```
## Loading required package: rpart
```

```r
pr.tree <-predict(model.tree,newdata=testing)
cm.tree <-confusionMatrix(pr.tree,testing$classe)
```

In terms of accuracy, for each model we have:
-**Logistic Regression model:** 0.9238
-**SVM model**: 0.9387
-**CART model**:0.4962

The tree shows reduced accuracy compared to the first two models.

## Model evaluation

Evaluate the three models and generate the comparison table accuracy.  
        

```r
#valid1 <-predict(model.treebag,valid)
validlogregboot <-predict(model.logregboot,valid)
validsvm <-predict(model.svm,valid)
validtree <-predict(model.tree,valid)

summa <-data.frame(validlogregboot,validsvm,validtree)
summa
```

```
##    validlogregboot validsvm validtree
## 1             <NA>        B         C
## 2                A        A         A
## 3                B        C         C
## 4                A        A         A
## 5                A        A         A
## 6                E        E         C
## 7                D        D         C
## 8                D        B         A
## 9                A        A         A
## 10               A        A         A
## 11            <NA>        A         C
## 12               C        C         C
## 13               B        B         C
## 14               A        A         A
## 15               E        E         C
## 16               E        E         A
## 17               A        A         A
## 18               D        B         A
## 19               B        B         A
## 20               B        B         C
```

## Results 

We see that except from problem_ids 3,8 and 18, the choice of the correct class is pretty straight forward. 
Going with the SVM results, the submission reached 19 out of 20 correct prediction.    
