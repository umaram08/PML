if(!file.exists('pml-training.csv')){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                destfile='pml-training.csv') }
if(file.exists('pml-training.csv')){
  training <- read.csv('pml-training.csv', na.strings = c(""," ","NA"), header = TRUE) }
str(training)


if(!file.exists('pml-testing.csv')){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                destfile='pml-testing.csv') }
if(file.exists('pml-testing.csv')){
  testing <- read.csv('pml-testing.csv', na.strings = c(""," ","NA"), header = TRUE) }
str(testing)

library(AppliedPredictiveModeling)
library(caret)
library(randomForest)
library(rattle)
library(rpart.plot)
library(kernlab)

ind <- which(is.na(pmatch(names(training), names(testing))))
names(training)[ind]
names(testing)[ind]

na_count <-sapply(training, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
nonzeroind <- which(na_count == 0)
nonzeroind <- nonzeroind[8:length(nonzeroind)]

training <- as.data.frame(training[,nonzeroind], drop = FALSE)
names(training)

testing <- as.data.frame(testing[,nonzeroind], drop = FALSE)
names(testing)

nsv <- nearZeroVar(training, saveMetrics = TRUE)
nsv

set.seed(32343)
partition <- createDataPartition(y = training$classe, p = 0.25, list = FALSE)
part1 <- training[partition,]
rest<- training[-partition,]
set.seed(32343)
partition <- createDataPartition(y = rest$classe, p = 0.33, list = FALSE)
part2 <- rest[partition,]
rest <- rest[-partition,]
set.seed(32343)
partition <- createDataPartition(y = rest$classe, p = 0.5, list = FALSE)
part3 <- rest[partition,]
part4 <- rest[-partition,]

set.seed(32343)
inTrain <- createDataPartition(y = part1$classe, p = 0.6, list = FALSE)
part1.train <- part1[inTrain,]
part1.test <- part1[-inTrain,]
set.seed(32343)
inTrain <- createDataPartition(y = part2$classe, p = 0.6, list = FALSE)
part2.train <- part2[inTrain,]
part2.test <- part2[-inTrain,]
set.seed(32343)
inTrain <- createDataPartition(y = part3$classe, p = 0.6, list = FALSE)
part3.train <- part3[inTrain,]
part3.test <- part3[-inTrain,]
set.seed(32343)
inTrain <- createDataPartition(y = part4$classe, p = 0.6, list = FALSE)
part4.train <- part4[inTrain,]
part4.test <- part4[-inTrain,]

set.seed(32343)
part1.modFit <- train(part1.train$classe ~., method = "rpart", data = part1.train, 
                      trControl = trainControl(method = "cv", number = 3))
fancyRpartPlot(part1.modFit$finalModel)
part1.pred <- predict(part1.modFit, part1.test)
confusionMatrix(part1.pred,part1.test$classe)$overall['Accuracy']

set.seed(32343)
part1.modFit <- train(part1.train$classe ~., method = "rf", data = part1.train, 
                      trControl = trainControl(method = "cv", number = 3))
part1.pred <- predict(part1.modFit, part1.test)
part1.accuracy <- confusionMatrix(part1.pred,part1.test$classe)$overall['Accuracy']
part1.accuracy
pred1 <- predict(part1.modFit, newdata = testing)
pred1

set.seed(32343)
part2.modFit <- train(part2.train$classe ~., method = "rf", data = part2.train, 
                      trControl = trainControl(method = "cv", number = 3))
part2.pred <- predict(part2.modFit, part2.test)
part2.accuracy <- confusionMatrix(part2.pred,part2.test$classe)$overall['Accuracy']
part2.accuracy
pred2 <- predict(part2.modFit, newdata = testing)
pred2

set.seed(32343)
part3.modFit <- train(part3.train$classe ~., method = "rf", data = part3.train, 
                      trControl = trainControl(method = "cv", number = 3))
part3.pred <- predict(part3.modFit, part3.test)
part3.accuracy <- confusionMatrix(part3.pred,part3.test$classe)$overall['Accuracy']
part3.accuracy
pred3 <- predict(part3.modFit, newdata = testing)
pred3

set.seed(32343)
part4.modFit <- train(part4.train$classe ~., method = "rf", data = part4.train, 
                      trControl = trainControl(method = "cv", number = 3))
part4.pred <- predict(part4.modFit, part4.test)
part4.accuracy <- confusionMatrix(part4.pred,part4.test$classe)$overall['Accuracy']
part4.accuracy
pred4 <- predict(part4.modFit, newdata = testing)
pred4

pred <- data.frame(pred1,pred2,pred3,pred4)
names(pred) <- c("Data.part1", "Data.part2", "Data.part3", "Data.part4")
pred

Accuracy <- data.frame(part1.accuracy, part2.accuracy, part3.accuracy, part4.accuracy)
Accuracy <- round(Accuracy * 100)
OutofSampleError <- 100 - Accuracy
Accuracy <- paste(Accuracy,'%',sep = "")
OutofSampleError <- paste(OutofSampleError,'%',sep = "")

Accuracy
OutofSampleError