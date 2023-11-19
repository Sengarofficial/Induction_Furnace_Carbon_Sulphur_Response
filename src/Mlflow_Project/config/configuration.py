from typing import Optional
from pathlib import Path
from src.Mlflow_Project.constants import * 
from src.Mlflow_Project import logger 
from src.Mlflow_Project.utils.utility import FileOperations
from src.Mlflow_Project.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                                     DataTransformationConfig,
                                                     ModelTrainerConfig, ModelEvaluationConfig)


import os
from dotenv import load_dotenv
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


# configuration manager class , will read all yaml files 

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
        schema_filepath: str = SCHEMA_FILE_PATH
    ) -> None:
        self.config = FileOperations.read_yaml(config_filepath)
        self.params = FileOperations.read_yaml(params_filepath)
        self.schema = FileOperations.read_yaml(schema_filepath)

        FileOperations.create_directories([self.config.artifacts_root])


        # Log successful initialization
        logger.info("ConfigurationManager initialized successfully.")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.data_ingestion
            FileOperations.create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file
            )

            return data_ingestion_config
        except Exception as e:
            logger.error(f"Error in get_data_ingestion_config: {e}")
            raise e 
        

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config.data_validation
            schema = self.schema.COLUMNS

            FileOperations.create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                STATUS_FILE=config.STATUS_FILE,
                valid_data_dir=config.valid_data_dir,
                all_schema=schema,
            )

            return data_validation_config
        except Exception as e:
            logger.error(f"Error in get_data_validation_config: {e}")
            raise e
        

    def get_data_transformation_config(self) -> DataTransformationConfig:
            
        try:
            config = self.config.data_transformation

            FileOperations.create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                 root_dir = config.root_dir,
                 data_path = config.data_path,
                 scaler_name = config.scaler_name,

            )
            
            return data_transformation_config
        except Exception as e:
            logger.error(f"Error in get_data_validation_config: {e}")
            raise e
        

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.RandomForestRegressor
        schema = self.schema

        FileOperations.create_directories([config.root_dir])

        target_values = schema.get('TARGET_COLUMN', {})
        target_column = {
            'target_1': target_values.get('target_1'),
            'target_2': target_values.get('target_2')
        }

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            imputer_name = config.imputer_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
                target_column=target_column  # Assign target_1 and target_2 to target_column
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
            config = self.config.model_evaluation
            params = self.params.RandomForestRegressor
            


            FileOperations.create_directories([config.root_dir])


            model_evaluation_config = ModelEvaluationConfig(
                 root_dir = config.root_dir,
                 test_x = config.test_x,
                 test_y = config.test_y,
                 model_path = config.model_path,
                 all_params = params,
                 metric_file_name = config.metric_file_name,
                 mlflow_uri= MLFLOW_URI,
            )

             return model_evaluation_config




