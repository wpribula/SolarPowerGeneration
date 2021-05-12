library(tidyverse)
library(caret)


################################################################################
#                                                                              #
#                   LOAD DATA, SEPARATION and PREPROCESSING                    #
#                                                                              #
# data source: https://www.kaggle.com/vipulgote4/solar-power-generation        #
#                                                                              #
################################################################################
# Load data from CSF file
data <- read.csv("Solar-Dataset.csv")
# Checking is daylight average
mean(data$Power.Generated[data$Is.Daylight==FALSE])
sd(data$Power.Generated[data$Is.Daylight==FALSE])
# Mean and standard deviation are 0 for Is.Daylight == FALSE.
# So there is no reason to predict for this as there is no sunlight at night.
data <- data %>% select(-Is.Daylight)
# Replace all NA
data[is.na(data)] <- 0

# Satndardise values if needed
# z_std <- function(observed) {
# 
#   result <- (observed - mean(observed)) / sd(observed)
# }
# data <- data %>% mutate_all(list(~ z_std(.)))

# Divide to train and test
indexes <- createDataPartition(data$Power.Generated, times = 1, 0.2, list = FALSE)
validation <- data %>% slice(indexes)
train_dada <- data %>% slice(-indexes)

################################################################################
#                                                                              #
#                        FUNCTION FOR TRAINING MODELS                          #
#                                                                              #
################################################################################
train_models = function(final_teaining = FALSE){
  # Day of the month is not very important too month is not natural cycle, so
  # sunlight is not dependent on it.
  # Not removing for validation as needed in report
  
  #-----------------------------------------------------------------------------
  #            Prepare Train and Test data if it is not final training
  #-----------------------------------------------------------------------------
  # This is for training with splitting for cross-check
  if(!final_teaining){
    indexes <- createDataPartition(train_dada$Power.Generated, times = 1, 0.2, list = FALSE)
    test <- train_dada %>% select(-Day) %>% slice(indexes)
    train <- train_dada %>% select(-Day) %>% slice(-indexes)
  }
  # This is for final training without splitting for cross-check
  if(final_teaining){
    test = validation %>% select(-Day)
    train <- train_dada %>% select(-Day)
  }
    
  #-----------------------------------------------------------------------------
  #                       USING AVERAGE AS PREDICION
  #-----------------------------------------------------------------------------
  miu <- mean(train$Power.Generated)
  y_hat <- rep(miu, length(test$Power.Generated))
  RMSE <- data.frame(Cycle = pass,
                     Method = "Average", 
                     RMSE = RMSE(y_hat,test$Power.Generated),
                     fit_index = NA)
  
  #-----------------------------------------------------------------------------
  #             KNN - k-Nearest Neighbors
  #                 - Number of Neighbors (k, numeric) 
  #-----------------------------------------------------------------------------
  if("knn" %in% use_methods){
    cat("Method: knn, Pass", pass,"\n")
    fit_knn <- train(train[1:13], train$Power.Generated, method="knn",
                     tuneGrid = data.frame(k = seq(1, 20, 1)))
    plot(fit_knn)
    y_hat_knn <- predict(fit_knn$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "KNN", 
                                   RMSE = RMSE(y_hat_knn,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_knn
    fit_index <<- fit_index + 1
  }
  
  #-----------------------------------------------------------------------------
  #             GLM - Generalized Linear Model
  #-----------------------------------------------------------------------------
  if("glm" %in% use_methods){
    cat("Method: glm, Pass", pass,"\n")
    fit_glm <- train(train[1:13], train$Power.Generated, method="glm")
    y_hat_glm <- predict(fit_glm$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "glm", 
                                   RMSE = RMSE(y_hat_glm,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_glm
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_glm$finalModel)))
  }
  
  #-----------------------------------------------------------------------------
  #              treebag - Bagged CART
  #-----------------------------------------------------------------------------
  if("treebag" %in% use_methods){
    cat("Method: treebag, Pass" ,pass,"\n")
    fit_treebag <- train(train[1:13], train$Power.Generated, method="treebag")
    y_hat_treebag <- predict(fit_treebag$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "treebag", 
                                   RMSE = RMSE(y_hat_treebag,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_treebag
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_treebag$finalModel)))
  }
  
  #-----------------------------------------------------------------------------
  #              rf - Conditional Inference Tree
  #                    - 1 - P-Value Threshold (mincriterion, numeric) 
  #-----------------------------------------------------------------------------
  if("rf" %in% use_methods){
    cat("Method: rf, Pass", pass,"\n")
    fit_rf <- train(train[1:13], train$Power.Generated, method="rf",
                    tuneGrid=expand.grid(mtry=seq(0,6,1)))
    plot(fit_rf)
    y_hat_rf <- predict(fit_rf$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "rf",
                                   RMSE = RMSE(y_hat_rf,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_rf
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_rf$finalModel)))
  }

  #-----------------------------------------------------------------------------
  #              ctree - Random Forest
  #                    - Number of Randomly Selected Predictors (mtry, numeric)
  #-----------------------------------------------------------------------------
  if("ctree2" %in% use_methods){
    cat("Method: ctree2, Pass", pass,"\n")
    fit_ctree2 <- train(train[1:13], train$Power.Generated, method="ctree2",
                       tuneGrid=expand.grid(mincriterion=seq(0,1,0.2), maxdepth=seq(0,3,1)))
    plot(fit_ctree2)
    y_hat_ctree2 <- predict(fit_ctree2$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "ctree2",
                                   RMSE = RMSE(y_hat_ctree2,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_ctree2
  }
    fit_index <<- fit_index + 1

  #-----------------------------------------------------------------------------
  #              rpart - CART
  #                    - Complexity Parameter (cp, numeric)
  #-----------------------------------------------------------------------------
  if("rpart" %in% use_methods){
    cat("Method: rpart, Pass", pass,"\n")
    fit_rpart <- train(train[1:13], train$Power.Generated, method="rpart",
                       tuneGrid=expand.grid(cp=seq(0, 0.05, 0.005)))
    plot(fit_rpart)
    y_hat_rpart <- predict(fit_rpart$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "rpart",
                                   RMSE = RMSE(y_hat_rpart,test$Power.Generated),
                                   fit_index = fit_index))
    fit[fit_index] <<- fit_rpart
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_rpart$finalModel)))
  }

  #-----------------------------------------------------------------------------
  #              rpart2 - CART
  #                     - Max Tree Depth (maxdepth, numeric)
  #-----------------------------------------------------------------------------
  if("rpart2" %in% use_methods){
    cat("Method: rpart2, Pass", pass,"\n")
    fit_rpart2 <- train(train[1:13], train$Power.Generated, method="rpart2",
                       tuneGrid=expand.grid(maxdepth=seq(0, 10, 1)))
    plot(fit_rpart2)
    y_hat_rpart2 <- predict(fit_rpart2$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "rpart2",
                                   RMSE = RMSE(y_hat_rpart2,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_rpart2
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_rpart2$finalModel)))
  }

  #-----------------------------------------------------------------------------
  #              bridge - Bayesian Ridge Regression
  #-----------------------------------------------------------------------------
  if("bridge" %in% use_methods){
    cat("Method: bridge, Pass", pass,"\n")
    fit_bridge <- train(train[1:13], train$Power.Generated, method="bridge")
    y_hat_bridge <- predict(fit_bridge, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "bridge",
                                   RMSE = RMSE(y_hat_bridge,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_bridge
    fit_index <<- fit_index + 1
  }

  #-----------------------------------------------------------------------------
  #              ppr - Projection Pursuit Regression
  #                  - Number of Terms (nterms, numeric)
  #-----------------------------------------------------------------------------
  if("ppr" %in% use_methods){
    cat("Method: ppr, Pass", pass,"\n")
    fit_ppr <- train(train[1:13], train$Power.Generated, method="ppr",
                        tuneGrid=expand.grid(nterms=seq(0, 8, 1)))
    plot(fit_ppr)
    y_hat_ppr <- predict(fit_ppr$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "ppr",
                                   RMSE = RMSE(y_hat_ppr,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_ppr
    fit_index <<- fit_index + 1
  }

  #-----------------------------------------------------------------------------
  #              gaussprLinear - Gaussian Process
  #-----------------------------------------------------------------------------
  if("gaussprLinear" %in% use_methods){
    cat("Method: gaussprLinear, Pass", pass,"\n")
    fit_gaussprLinear <- train(train[1:13], train$Power.Generated, method="gaussprLinear")
    y_hat_gaussprLinear <- predict(fit_gaussprLinear, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "gaussprLinear",
                                   RMSE = RMSE(y_hat_gaussprLinear,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_gaussprLinear
    fit_index <<- fit_index + 1
  }

  #-----------------------------------------------------------------------------
  #              gamSpline - Generalized Additive Model using Splines
  #                        - Degrees of Freedom (df, numeric)
  #-----------------------------------------------------------------------------
  if("gamSpline" %in% use_methods){
    cat("Method: gamSpline, Pass", pass,"\n")
    fit_gamSpline <<- train(train[1:13], train$Power.Generated, method="gamSpline",
                     tuneGrid=expand.grid(df=seq(0, 16, 2)))
    plot(fit_gamSpline)
    y_hat_gamSpline <- predict(fit_gamSpline$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "gamSpline",
                                   RMSE = RMSE(y_hat_gamSpline,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_gamSpline
    fit_index <<- fit_index + 1
    importance <<- rbind(importance, as.data.frame(varImp(fit_gamSpline$finalModel)))
  }

  #-----------------------------------------------------------------------------
  #              brnn - Bayesian Regularized Neural Networks
  #                   - Number of Neurons (neurons, numeric)
  #-----------------------------------------------------------------------------
  if("brnn" %in% use_methods){
    cat("Method: brnn, Pass", pass,"\n")
    fit_brnn <- train(train[1:13], train$Power.Generated, method="brnn",
                     tuneGrid=expand.grid(neurons=seq(5, 10, 1)))
    plot(fit_brnn)
    y_hat_brnn <- predict(fit_brnn$finalModel, test[1:13])
    RMSE <- rbind(RMSE, data.frame(Cycle = pass,
                                   Method = "brnn",
                                   RMSE = RMSE(y_hat_brnn,test$Power.Generated),
                                   fit_index = fit_index))
    fit[[fit_index]] <<- fit_brnn
    fit_index <<- fit_index + 1
  }
  RMSE
}

################################################################################
#                                                                              #
#                       MAIN FUNCTION FOR PROCESSING ALL                       #
#                                                                              #
################################################################################
# specif which methods to use
use_methods = c("knn",
                "glm",
                "treebag",
                "rf",
                "ctree2",
                "rpart",
                "rpart2",
                "bridge",
                "ppr",
                "gaussprLinear",
                "gamSpline",
                "brnn")

# Initialize Results data frame - this is for all RMSEs calculated in all training runs
Results <- data.frame()
# Initialize list for fir models from train() functions - fit can't be stored in Results data frame
fit <- list()
# Index of fit model in fit list, which will be stored in Result data frame to link these together.
fit_index <- 1
# Initialize data frame for variables importance, it is stored for report purposes.
importance <- data.frame()

#-------------------------------------------------------------------------------
#                            >>> RUN TRAINING <<<
#-------------------------------------------------------------------------------
for (pass in 1:5){
  Results <- rbind(Results, train_models())
}
#-------------------------------------------------------------------------------
#                           >>> PROCESS RESULTS <<<
#-------------------------------------------------------------------------------
# Calculate average RMSE for all methods
RMSE_avg <- Results %>% group_by(Method) %>% summarise(RMSE_AVG = mean(RMSE))
# Extract methods ordered by RMSE
methods_ordered <- RMSE_avg %>% arrange(RMSE_AVG) %>% .$Method
# Extract best methods and adjust use_methods list
use_methods <- methods_ordered[6]

#-------------------------------------------------------------------------------
#                          >>> RUN FULL TRAINING <<<
#  - Run full training for best performing method(s) and all train data.
#-------------------------------------------------------------------------------
pass <- 0 # Set pass to 0 to recognize final results in Results data frame.
Results <- rbind(Results, train_models(final_teaining = TRUE))

#-------------------------------------------------------------------------------
#                        >>> PROCESS FULL TRAINING <<<
#-------------------------------------------------------------------------------
# Pick up final models form Results.
final_models <- Results %>% filter(Cycle == 0 & Method != "Average")
# Get fit models for full training methods from fit list and calculate y_hat
for (i in 1:nrow(final_models)){
  current_y_hat <- predict(fit[final_models[i,]$fit_index], validation)
  if (i == 1){
    y_hat <- data.frame(current_y_hat)
  }
  else{
    y_hat <- y_hat %>% cbind(data.frame(current_y_hat))
  }
}
# Calculate mean for all models y_hats
colnames(y_hat) <- c(1:nrow(final_models))
y_hat <- rowMeans(y_hat)
# Calculate RMSE for final y_hat
RMSE <- RMSE(y_hat, validation$Power.Generated)
# Save environment for report
save.image('Environment.rdata')


