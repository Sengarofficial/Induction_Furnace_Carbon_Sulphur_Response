# Updating our components 

import os 
import urllib.request as request
from src.Mlflow_Project import logger 
from src.Mlflow_Project.utils.utility import FileOperations
from src.Mlflow_Project.entity.config_entity import DataIngestionConfig
from typing import Optional

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, download_timeout: Optional[int] = 60):
        self.config = config
        self.download_timeout = download_timeout

    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.info(f"Downloading file from {self.config.source_URL} to {self.config.local_data_file}")

                # Set a timeout for the download request
                response = request.urlopen(self.config.source_URL, timeout=self.download_timeout)

                with open(self.config.local_data_file, 'wb') as file:
                    file.write(response.read())

                logger.info(f"File downloaded successfully.")
            else:
                logger.info(f"File already exists at {self.config.local_data_file}")
        except Exception as e:
            logger.error(f"An error occurred while downloading the file: {e}")
            raise e

