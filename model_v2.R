library(caret)

#### LOAD DATA ###
training <- read.csv("./input/pml-training.csv", na.string=c("","#DIV/0!","NA"))
testing <- read.csv("./input/pml-testing.csv", na.string=c("","#DIV/0!","NA"))


#### SET OUTCOME AND PROBLEM_ID COLUMNS TO FIRST COLUMN ####
training <- training[,c(160,1:159)]
testing <- testing[,c(160,1:159)]


#### DROP TECHINAL COLUMNS ####
selected <- !(colnames(training) %in% c("X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window","new_window"))
training <- training[,selected]
testing <- testing[,selected]


#### SET NUMERIC COLUMNS TO NUMERIC CLASS ####
numeric_columns <- !(colnames(training) %in% c("classe","user_name","problem_id"))
training[,numeric_columns] <- sapply(training[,numeric_columns],as.numeric)
testing[,numeric_columns] <- sapply(testing[,numeric_columns],as.numeric)


#### DISCARD NEAR ZERO VARIANCE PREDICTORS ####
nzv <- nearZeroVar(training,saveMetrics = T)
training <- training[,!nzv$nzv]
testing <- testing[,!nzv$nzv]


#### IMPUTE NA'S ####
numeric_columns <- !(colnames(training) %in% c("classe","user_name","problem_id"))
imputation <- preProcess(training[,numeric_columns],method = "knnImpute")
training[,numeric_columns] <- predict(imputation,training[,numeric_columns])
testing[,numeric_columns] <- predict(imputation,testing[,numeric_columns])


#### CREATE DUMMY VARIABLES ####
dummies <- dummyVars(~., data = training[,-1])
training <- data.frame(classe = training$classe, predict(dummies,training))
testing <- data.frame(problem_id = testing$problem_id, predict(dummies,testing))


#### SLICE DATA ####
inTrain <- createDataPartition(training$classe, p = 0.7, list = F)
crossvalid <- training[-inTrain,]
training <- training[inTrain,]


#### DEFINE TRAINING CONTROL ####
trainControl <- trainControl(method = "repeatedcv",
                             number = 2,
                             repeats = 1,
                             preProcOptions = list(thresh = 0.95),
                             verboseIter = TRUE)

#### TRAIN RANDOM FOREST MODEL ####
modelFitRF <- train(training$classe~.,method = "rf", 
                  data = training, preProcess = "pca", 
                  trControl = trainControl)


plot(modelFitRF)
confusionMatrix(predict(modelFitRF,crossvalid),crossvalid$classe)
plot(varImp(modelFitRF))

#### TRAIN SVM MODEL ####
modelFitSVM <- train(training$classe~.,method = "svmRadial", 
                  data = training, preProcess = "pca", 
                  trControl = trainControl)
plot(modelFitSVM)
confusionMatrix(predict(modelFitSVM,crossvalid),crossvalid$classe)



#### TRAIN GBM MODEL ####
modelFitGBM <- train(training$classe~.,method = "gbm", 
                      data = training, preProcess = "pca", 
                      trControl = trainControl, verbose = F)
plot(modelFitGBM)
confusionMatrix(predict(modelFitGBM,crossvalid),crossvalid$classe)



resamp <- resamples(list(RF = modelFitRF, SVM = modelFitSVM, NNET = modelFitNNET, GBM = modelFitGBM))
summary(resamp)
bwplot(resamp)
xyplot(resamp)




pca <- preProcess(training[,-1],method = "pca",thresh = 0.95)
pcomps <- predict(pca,training[,-1])

