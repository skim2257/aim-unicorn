library(mRMRe)
library(dplyr)
library(dotenv)
library(survcomp)
library(caret)


# prepares joined RADCURE dataframe
get_data <- function(df_f_path, 
                     df_c_path, 
                     split = "training",
                     path = NULL) {
    # if path is not null and file exists, return the dataframe
    if (!is.null(path)) {
        if (file.exists(path)) {
            print("File exists! Loading...")
            df <- read.csv(path) %>%
                mutate(death = as.numeric(death)) %>%
                mutate(survival_time = as.numeric(survival_time))

            return(as.data.frame(df)) 
        }
    }
    df_f <- read.csv(df_f_path) %>%
        mutate(patient_id = sapply(strsplit(image_path, "/"), function(x) sub(".nii.gz", "", tail(strsplit(x[length(x)], "_")[[1]], 1))))
    
    df_c <- read.csv(df_c_path) %>%
        filter(`RADCURE.challenge` == split) %>%
        mutate(death = as.numeric(death))
    
    # Perform the inner join
    df <- merge(df_f, df_c, by.x = "patient_id", by.y = "ID")

    # Select the desired columns
    df <- df[, c("patient_id", "survival_time", "death", grep("^pred", names(df), value = TRUE))]

    # Set the row names to patient_id and remove the patient_id column
    rownames(df) <- df$ID
    df$ID <- NULL

    # save the dataframe to path if path is not null
    if (!is.null(path)) {
        write.csv(df, path, row.names = FALSE)
    }
    return(df)

}

# removes patient_id and survival_time columns from clinical sheet to prepare for modeling
prepare_features <- function(fmcib_df) {
    return (fmcib_df %>% 
                select(-patient_id, -survival_time) %>%
                as.data.frame())
}

# Trains mRMR model and returns the selected features indices
run_mrmr <- function(fmcib_df, 
                     n_features = 30) {
    dd                            <- mRMR.data(data=fmcib_df)
    fmcib_results                 <- mRMR.classic(data = dd,   
                                                  target_indices = c(1),
                                                  feature_count = n_features)

    # these are the solutions!
    fmcib_indices                 <- solutions(fmcib_results)

    return(fmcib_indices[[1]])
}

train <- function(fmcib_train, n_features) {
    fmcib_features                <- prepare_features(fmcib_train)
    mrmr_idx                      <- run_mrmr(fmcib_features, n_features=n_features)

    if (length(mrmr_idx) == 1) {
        mrmr_idx <- c(mrmr_idx) 
    }
    # grab the best features
    fmcib_train_X                 <- fmcib_features[,c(mrmr_idx)]

    # adding endpoints back into the dataframe
    fmcib_train_X$survival_time   <- as.numeric(fmcib_train$survival_time)
    fmcib_train_X$death           <- as.numeric(fmcib_train$death)

    # train the CoxPH model
    print(paste("Training CoxPH model with", n_features, "features..."))
    model                         <- coxph(Surv(survival_time, death) ~ .,    
                                           x=TRUE, 
                                           y=TRUE, 
                                           method="breslow", 
                                           data=fmcib_train_X)
    return(list(mrmr_idx = mrmr_idx, model = model))    

}

main <- function() {
    print("Starting...")
    
    load_dot_env()
    set.thread.count(16)

    # get fmcib features (df_f) and clinical data (df_c)
    df_f_path                     <- Sys.getenv("TRAIN_PATH")
    df_c_path                     <- Sys.getenv("SURV_PATH")

    # get train and test sets
    fmcib_train                   <- get_data(df_f_path, df_c_path, split = "training", path="./data/fmcib_train.csv")
    fmcib_test                    <- get_data(df_f_path, df_c_path, split = "test", path="./data/fmcib_test.csv")

    # sanity check, output dimensions of train and test
    print("Data loaded!...")
    print(paste("train dim:", paste(dim(fmcib_train), collapse = " ")))
    print(paste("test dim:", paste(dim(fmcib_test), collapse = " ")))

    # grab features + run mRMR
    print("Preparing features...")
    
    # create folds
    set.seed(42)
    folds                         <- createFolds(fmcib_train$patient_id, k = 5)
    ci_k                          <- list()
    for (i in 1:5) {
        fmcib_train_fold          <- fmcib_train[folds[[i]],]
        fmcib_val_fold            <- fmcib_train[-folds[[i]],]

        # prepare features for this fold
        # train_X                   <- prepare_features(fmcib_train_fold)
        val_X                     <- prepare_features(fmcib_val_fold)

        best_k                    <- 0
        best_ci                   <- 0
        for (k in seq(2, 102, by=4)) {
            # train the CoxPH model for this fold
            model                 <- train(fmcib_train_fold, n_features=k)$model

            # get concordance index with 95% confidence interval and p-value
            val_y_hat             <- predict(model, newdata=val_X, type="risk")
            
            cindex                <- concordance.index(val_y_hat, 
                                                       fmcib_val_fold$survival_time, 
                                                       fmcib_val_fold$death,
                                                       method = "noether", 
                                                       alpha = 0.05, 
                                                       alternative = "two.sided")

            if(cindex$c.index > best_ci) {
                best_ci            <- cindex$c.index
                best_k             <- k

                print(paste("NEW BEST!! fold:", i, "k:", best_k, "c.index:", best_ci)) 
            }
        }
        ci_k[[i]]             <- best_k
        print(paste("\n\nBest k for fold", i, ":", best_k))
    }
    top_k                          <- ci_k[which.max(unlist(ci_k))]
    

    print(paste("Training FINAL model with", top_k, "features..."))
    # prepare test data for inference
    results                        <- train(fmcib_train, n_features=as.numeric(top_k))
    model_top                      <- results$model
    mrmr_idx_top                   <- results$mrmr_idx

    fmcib_test_X                   <- prepare_features(fmcib_test)[,c(mrmr_idx_top)]
    print("Running inference...")
    # CoxPH inference
    fmcib_test_y_hat               <- predict(model_top, newdata=fmcib_test_X, type="risk")

    # get concordance index with 95% confidence interval and p-value
    cindex <- concordance.index(fmcib_test_y_hat, 
                                fmcib_test$survival_time, 
                                fmcib_test$death,
                                method = "noether", 
                                alpha = 0.05, 
                                alternative = "two.sided")

    print(paste("C-index on test set:", cindex$c.index))
    print(paste("Confidence interval:", cindex$lower, ",", cindex$upper))
    print(paste("p-value:", cindex$p.value))
    print("Done!!")
}

main()