import os
import pandas as pd
import numpy as np
from src.Mlflow_Project.__init__ import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.Mlflow_Project.entity.config_entity import DataTransformationConfig
import joblib




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    ## Note: You can add different data transformtion techniques such as Scaler, PCA and all 
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model 

    # defining train test split method 

    def train_test_split(self):
        df = pd.read_csv(self.config.data_path)

        conversion_factor = 1000 

        # List of columns toconvert 

        columns_to_convert = ['SPONGE_IRON_MT', 'SCRAP_MT', 'MT_PRODUCTION_PER_HEAT']

        # Convert the specified columns from metric tons to kilograms
        df[columns_to_convert] = df[columns_to_convert].apply(lambda x: x * conversion_factor)

        # We need to drop unnecessary columns like AVG_POWER_PER_HEAT_KW, POWER_CONSUMED_KWH, Manganese(Mn%) etc

        columns_to_drop = ['AVG_POWER_PER_HEAT_KW', 'POWER_CONSUMED_KWH', 'Manganese(Mn%)', 'Silicon(Si%)', 'Phosphorous(P%)', 'MT_PRODUCTION_PER_HEAT']

        # Drop the specified columns
        df = df.drop(columns=columns_to_drop, axis=1)  # axis=1 indicates columns
        

        train, test = train_test_split(df, test_size=0.25, random_state=42)  

        # Separate the target variable (RUL) from the input features in the training and test datasets
        X_train = train.drop(columns=["Carbon(C%)", "Sulphur(S%)"])
        y_train = train[["Carbon(C%)", "Sulphur(S%)"]]
        X_test = test.drop(columns=["Carbon(C%)", "Sulphur(S%)"])
        y_test = test[["Carbon(C%)", "Sulphur(S%)"]]

        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info("Train scaling done..!!")

        # Transform test data using the SAME scaler fitted on the training data
        X_test_scaled = scaler.transform(X_test)
        logger.info("Test data transform done...!!")

        train_scaled_df = pd.DataFrame(X_train_scaled, columns= X_train.columns)
        test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns)

        # Adding new column RUL to the scaled datasets 
        train_scaled_df[["Carbon(C%)", "Sulphur(S%)"]] = y_train
        test_scaled_df[["Carbon(C%)", "Sulphur(S%)"]] = y_test

        

        # Save the scaled datasets and target variables
        train_scaled_df.to_csv(os.path.join(self.config.root_dir, "train_scaled.csv"), index=False)
        test_scaled_df.to_csv(os.path.join(self.config.root_dir, "test_scaled.csv"), index=False)


        joblib.dump(scaler, os.path.join(self.config.root_dir, self.config.scaler_name))

        
        logger.info("Splitted data into train and test")
        logger.info(train.shape)
        logger.info(test.shape)


        print(train.shape)
        print(test.shape)

        
        

        
        