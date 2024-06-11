library(mRMRe)
library(dplyr)
library(dotenv)
library(survcomp)

# prepares joined RADCURE dataframe
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

# removes patient_id and survival_time columns from clinical sheet to prepare for modeling
prepare_features <- function(fmcib_df) {
    return (fmcib_df %>% select(-patient_id, -survival_time))
}

# 
run_mrmr <- function(fmcib_df, mRMR_fitter) {
    dd                            <- mRMR.data(data=fmcib_df)
    fmcib_classic                 <- mRMR_fitter(data = dd,   
                                                  target_indices = c(1),
                                                  feature_count = 30)
    fmcib_indices                 <- solutions(fmcib_classic)
    
    return(fmcib_indices[[1]])
}

main <- function() {
    print("Starting...")
    load_dot_env()
    set.thread.count(8)

    # get fmcib features (df_f) and clinical data (df_c)
    df_f                          <- read.csv(Sys.getenv("TRAIN_PATH"))
    df_c                          <- read.csv(Sys.getenv("SURV_PATH"))

    # get train and test sets
    fmcib_train                   <- get_data(df_f, df_c, split = "training")
    fmcib_test                    <- get_data(df_f, df_c, split = "test")

    # sanity check, output dimensions of train and test
    print("Data loaded!...")
    print(paste("train dim:", paste(dim(fmcib_train), collapse = " ")))
    print(paste("test dim:", paste(dim(fmcib_test), collapse = " ")))

    # grab features + run mRMR
    print("Preparing features...")
    fmcib_all_feat                <- prepare_features(fmcib_train)
    mrmr_idx                      <- run_mrmr(fmcib_all_feat, mRMR.classic)

    # grab the best features
    fmcib_train_X                 <- fmcib_all_feat[,c(mrmr_idx)]

    # adding endpoints back into the dataframe
    fmcib_train_X$survival_time   <- fmcib_train$survival_time
    fmcib_train_X$death           <- fmcib_train$death

    # train the CoxPH model
    print("Training CoxPH model...")
    model                         <- coxph(Surv(survival_time, death) ~ .,    
                                           x=TRUE, 
                                           y=TRUE, 
                                           method="breslow", 
                                           data=fmcib_train_X)
    print(summary(model))

    # prepare test data for inference
    fmcib_test_X                  <- prepare_features(fmcib_test)[,c(mrmr_idx)]

    print("Running inference...")
    # CoxPH inference
    fmcib_y_hat                   <- predict(model, newdata=fmcib_test_X, type="risk")

    # get concordance index with 95K% CI and stuff
    cindex <- concordance.index(fmcib_y_hat, 
                                fmcib_test$survival_time, 
                                fmcib_test$death,
                                method = "noether", 
                                alpha = 0.05, 
                                alternative = "two.sided")
    print(cindex)
    print("Done!!")
}

main()