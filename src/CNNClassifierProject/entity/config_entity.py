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
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    param_freeze_layers: bool
