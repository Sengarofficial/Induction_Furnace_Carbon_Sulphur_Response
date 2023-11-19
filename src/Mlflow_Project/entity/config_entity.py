from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen = True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    valid_data_dir: Path
    all_schema: dict 



@dataclass(frozen = True)  
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path
    scaler_name: str 


@dataclass(frozen = True)
class ModelTrainerConfig:
    root_dir: Path 
    train_data_path: Path 
    test_data_path: Path
    model_name: str
    imputer_name: str
    n_estimators: float
    max_depth: float
    target_column: str  


@dataclass(frozen = True)
class ModelEvaluationConfig:
    root_dir: Path
    test_x: Path
    test_y: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    mlflow_uri: str

