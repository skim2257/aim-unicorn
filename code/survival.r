library(mRMRe)
library(dplyr)
library(dotenv)

load_dot_env()
set.thread.count(8)

get_data <- function(df_f, df_c, split = "training") {
    df_f <- df_f %>%
        mutate(patient_id = sapply(strsplit(image_path, "/"), function(x) sub(".nii.gz", "", tail(strsplit(x[length(x)], "_")[[1]], 1))))
    
    df_c <- df_c %>%
        filter(`RADCURE.challenge` == split) %>%
        mutate(death = as.numeric(death))
    
    # Perform the inner join
    df <- merge(df_f, df_c, by.x = "patient_id", by.y = "ID")

    # Select the desired columns
    df <- df[, c("patient_id", "survival_time", "death", grep("^pred", names(df), value = TRUE))]

    # Set the row names to patient_id and remove the patient_id column
    rownames(df) <- df$ID
    df$ID <- NULL
  
  return(df)
}

args <- list(train_features_path = Sys.getenv("TRAIN_PATH"), 
             clinical_path = Sys.getenv("SURV_PATH"))

# get fmcib features (df_f) and clinical data (df_c)
df_f <- read.csv(args$train_features_path)
df_c <- read.csv(args$clinical_path)

# get train and test sets
fmcib_train    <- get_data(df_f, df_c, split = "training")
fmcib_test     <- get_data(df_f, df_c, split = "test")

# sanity check, output dimensions of train and test
print(paste("train dim:", paste(dim(fmcib_train), collapse = " ")))
print(paste("test dim:", paste(dim(fmcib_test), collapse = " ")))

# grab only the features + death column
features_only <- subset(fmcib_train, select = -c(patient_id, survival_time))
dd <- mRMR.data(data = features_only)

# run the classic method
fmcib_classic <- mRMR.classic(data = dd, 
                              target_indices = c(1),
                              feature_count = 30)

# get the indices of the best features
fmcib_indices <- solutions(fmcib_classic)

# grab the best features
fmcib_mrmr    <- features_only[,c(fmcib_indices[[1]])]
print(paste("good features:", fmcib_indices))
print(dim(fmcib_mrmr))

# adding endpoints back into the dataframe
fmcib_mrmr$survival_time <- fmcib_train$survival_time
fmcib_mrmr$death         <- fmcib_train$death

# train the CoxPH model
model <- coxph(Surv(survival_time, death) ~ .,
                 x=TRUE, y=TRUE, method="breslow", data=fmcib_mrmr)
print(summary(model))

# prepare test data for inference
fmcib_test_features <- subset(fmcib_test, select = -c(patient_id, survival_time))
fmcib_test_features <- fmcib_test_features[,c(fmcib_indices[[1]])]

# get the predictions
fmcib_y_hat         <- predict(model, newdata=fmcib_test_features, type="risk")

# import survcomp + get concordance index
library(survcomp)
cindex <- concordance.index(fmcib_y_hat, 
                            fmcib_test$survival_time, 
                            fmcib_test$death,
                            method = "noether", 
                            alpha = 0.05, 
                            alternative = "two.sided")
print(cindex)
