# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data import FLIPPreprocess
from bionemo.data.metrics import accuracy, mse, per_token_accuracy
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.utils import (
    setup_trainer,
)
from lightning.pytorch.loggers import WandbLogger
import wandb

@hydra_runner(config_path="../esm1nv/conf", config_name="downstream_flip_sec_str")  # ESM1
def main(cfg) -> None:
    logging.info("\n\n************* Finetune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    
    if cfg.wandb_artifacts.wandb_use_artifacts_data:
        artifact_data_dir = WandbLogger.download_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_data_path)
        cfg.model.data.dataset_path=artifact_data_dir+cfg.model.data.dataset_path
        print(cfg.model.data.dataset_path)
    if cfg.wandb_artifacts.wandb_use_artifacts_model:
        artifact_model_dir = WandbLogger.download_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_model_path)
        print(artifact_model_dir)
        print(artifact_model_dir+cfg.model.restore_encoder_path)
        cfg.model.restore_encoder_path=artifact_model_dir+cfg.model.restore_encoder_path
        
    # Do preprocessing if preprocess
    if cfg.do_preprocessing:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_all_datasets(output_dir=cfg.model.data.preprocessed_data_path)
        

    # Load model
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg
    trainer = setup_trainer(cfg, builder=None)
    
    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = FineTuneProteinModel.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = FineTuneProteinModel(cfg.model, trainer)

    metrics = {}
    metrics_args = {}
    for idx, name in enumerate(cfg.model.data.target_column):
        if cfg.model.data.task_type == "token-level-classification":
            metrics[name + "_accuracy"] = per_token_accuracy
            metrics_args[name + "_accuracy"] = {"label_id": idx}
        elif cfg.model.data.task_type == "classification":
            metrics[name + "_accuracy"] = accuracy
            metrics_args[name + "_accuracy"] = {}
        elif cfg.model.data.task_type == "regression":
            metrics[name + "_MSE"] = mse
            metrics_args[name + "_MSE"] = {}

    model.add_metrics(metrics=metrics, metrics_args=metrics_args)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        if "test" in cfg.model.data.dataset:
            trainer.limit_train_batches = 0
            trainer.limit_val_batches = 0
            trainer.fit(model)
            trainer.test(model, ckpt_path=None)
        else:
            raise UserWarning(
                "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
            )
        logging.info("************** Finished Testing ***********")
    
    
    if cfg.wandb_artifacts.wandb_use_artifacts_data:
        wandb_logger = WandbLogger(project=cfg.exp_manager.wandb_logger_kwargs.project, 
                                       name=cfg.exp_manager.wandb_logger_kwargs.name)
        wandb_logger.use_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_data_path)
    if cfg.wandb_artifacts.wandb_use_artifacts_model:
        wandb_logger = WandbLogger(project=cfg.exp_manager.wandb_logger_kwargs.project, 
                                       name=cfg.exp_manager.wandb_logger_kwargs.name)
        wandb_logger.use_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_model_path)
    
    wandb.finish()
    

if __name__ == '__main__':
    main()
