from src.Mlflow_Project.config.configuration import ConfigurationManager
from src.Mlflow_Project.components.model_trainer import ModelTrainer
from src.Mlflow_Project import logger 

STAGE_NAME = "Model Training Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass 

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config = model_trainer_config)
            model_trainer_config.train()

        except Exception as e:
            logger.exception(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)



        