# Updating our components 

import os 
import urllib.request as request
from src.Mlflow_Project import logger 
from src.Mlflow_Project.utils.utility import FileOperations
from src.Mlflow_Project.entity.config_entity import (DataIngestionConfig, DataValidationConfig)
from typing import Optional
import pandas as pd
from typing import Optional



class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.valid_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    self.write_validation_status(validation_status)
                    break
                else:
                    validation_status = True

            # Move the writing of the validation status outside the loop
            if validation_status is not None:
                self.write_validation_status(validation_status)

            return validation_status

        except Exception as e:
            logger.error(f"Error in validate_all_columns: {e}")
            raise e

    def write_validation_status(self, validation_status: bool) -> None:
        try:
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")
        except Exception as e:
            logger.error(f"Error writing validation status: {e}")
            raise e
