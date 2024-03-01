from dataclasses import dataclass, field
from pathlib import Path

#normally the values of the entities are the same values 
# found in the configuration yaml file
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_dir: Path
    updated_base_model_dir: Path
    params_image_size: list
    params_include_top: bool
    params_weights: str
    params_classes: int
    param_freeze_layers: bool


@dataclass(frozen=True)
class TrainingConfigs:
    root_dir: Path
    model_dir: Path
    updated_base_model_dir: Path
    training_csv_dir: Path
    training_data_dir: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_base_update_model: Path
    path_of_trained_model: Path
    training_data: Path
    all_params:dict
    params_image_size: list
    params_batch_size: int

    
@dataclass(frozen=True)
class PredictConfig:
    path_to_model:Path
    path_to_weights:Path
    params_image_size: list
    
