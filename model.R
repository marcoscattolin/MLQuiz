library(caret)

#### LOAD DATA ###
training <- read.csv("./input/pml-training.csv",na.string=c("","#DIV/0!","NA"))
testing <- read.csv("./input/pml-testing.csv",na.string=c("","#DIV/0!","NA"))



#### EXTRACT OUTCOMES IN DIFFERENT DATAFRAME ####
classe <- training$classe
problem_id <- testing$problem_id
training <- training[,-160]
testing <- testing[,-160]

#### DISCARD TECHINAL COLUMNS ####
selected <- !(colnames(training) %in% c("X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window","new_window"))
training <- training[,selected]
testing <- testing[,selected]



#### DISCARD NEAR ZERO VARIANCE PREDICTORS ####
nzv <- nearZeroVar(training,saveMetrics = T)
training <- training[,!nzv$nzv]
testing <- testing[,!nzv$nzv]


#### SET RETAINED COLUMNS TO NUM ####
training[,-1] <- sapply(training[,-1],as.numeric)
testing[,-1] <- sapply(testing[,-1],as.numeric)

#### IMPUTE NA'S ####
imputation <- preProcess(training[,-1],method = "knnImpute")
training[,-1] <- predict(imputation,training[,-1])
testing[,-1] <- predict(imputation,testing[,-1])


#### CREATE DUMMY VARIABLES ####
dummies <- dummyVars(~., data = training)
training <- data.frame(predict(dummies,training))
testing <- data.frame(predict(dummies,testing))


# #### ADD BACK OUTCOME ####
training <- data.frame(classe = classe,training)
testing <- data.frame(problem_id = problem_id,testing)


#### SPLIT TRAINING MODEL ####
inTrain <- createDataPartition(classe,p = 0.7,list = F)
training <- training[inTrain,]
crossvalidation <- training[-inTrain,]
# trainingoutcome <- outcome[inTrain]
# crossvalidationoutcome <- outcome[-inTrain]

#### CALCULATE PRINCIPAL COMPONENTS ####
pca <- preProcess(training[,-1],method = "pca",thresh = 0.9)
trainingPC <- predict(pca,training[,-1])
crossvalidationPC <- predict(pca,crossvalidation[,-1])


#### TRAIN MODEL ####
Sys.time()
modelFit <- train(training$classe~.,method = "rf",data = trainingPC)
Sys.time()

res <- predict(modelFit,crossvalidationPC)



#### 10-FOLD RANDO FOREST ####
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1, preProcOptions = list(thresh = 0.95))
Sys.time()
modelFitFolded <- train(training$classe~.,method = "rf", data = training, preProcess = "pca", trControl = fitControl)
Sys.time()

resFolded <- predict(modelFitFolded,crossvalidationPC)


save(modelFit,file="singleFoldModel.Rdata")

