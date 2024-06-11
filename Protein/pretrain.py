# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks
from bionemo.data import FLIPPreprocess
from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector

from lightning.pytorch.loggers import WandbLogger

import wandb


@hydra_runner(config_path="conf", config_name="pretrain_esm2_8M")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
        
    

    if cfg.do_training:
        if cfg.wandb_artifacts.wandb_use_artifacts:
            artifact_dir = WandbLogger.download_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_path)
            cfg.model.data.dataset_path=artifact_dir+cfg.model.data.dataset_path
            cfg.model.data.uf90.uniref90_path=artifact_dir+cfg.model.data.uf90.uniref90_path

        callbacks = setup_dwnstr_task_validation_callbacks(cfg)
        
        trainer = setup_trainer(cfg, callbacks=callbacks)
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = esm1nv_model.ESM2nvModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
        trainer.fit(model)
        if cfg.wandb_artifacts.wandb_use_artifacts:
            wandb_logger = WandbLogger(project=cfg.exp_manager.wandb_logger_kwargs.project, 
                                       name=cfg.exp_manager.wandb_logger_kwargs.name)
            wandb_logger.use_artifact(artifact=cfg.wandb_artifacts.wandb_use_artifact_path)
        if cfg.exp_manager.create_wandb_logger:
            wandb.finish()
        
        logging.info("************** Finished Training ***********") 
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = ESM2Preprocess()
        
        if cfg.exp_manager.create_wandb_logger:
            run = wandb.init(project=cfg.exp_manager.wandb_logger_kwargs.project,
                            name=cfg.exp_manager.wandb_logger_kwargs.name,
                            job_type=cfg.exp_manager.wandb_logger_kwargs.job_type,
                            config=OmegaConf.to_container(cfg))
        
        if cfg.wandb_artifacts.wandb_use_artifacts:
            artifact = run.use_artifact(cfg.wandb_artifacts.wandb_use_artifact_path, type='dataset')
            artifact_folder = artifact.download()
            
            train_uf50_datapath = artifact_folder+cfg.model.data.uf50_datapath
            train_uf90_datapath = artifact_folder+cfg.model.data.uf90_datapath
            train_cluster_mapping_tsv = artifact_folder+cfg.model.data.cluster_mapping_tsv
        else:
            train_uf50_datapath = cfg.model.data.train.uf50_datapath
            train_uf90_datapath = cfg.model.data.train.uf90_datapath
            train_cluster_mapping_tsv = cfg.model.data.train.cluster_mapping_tsv

        if not os.path.exists(train_uf50_datapath):
            print(train_uf50_datapath)
            raise FileNotFoundError(
                "input argument ++cfg.model.data.train.uf50_datapath: {cfg.model.data.train.uf50_datapath} is not found."
            )
        if not os.path.exists(train_uf90_datapath):
            raise FileNotFoundError(
                "input argument ++cfg.model.data.train.uf90_datapath: {cfg.model.data.train.uf90_datapath} is not found."
            )
        if not os.path.exists(train_cluster_mapping_tsv):
            raise FileNotFoundError(
                "input argument ++cfg.model.data.train.cluster_mapping_tsv: {cfg.model.data.train.cluster_mapping_tsv} is not found."
            )

        preprocessor.prepare_dataset(
            uf50_datapath=train_uf50_datapath,
            uf90_datapath=train_uf90_datapath,
            cluster_mapping_tsv=train_cluster_mapping_tsv,
            uf50_output_dir=cfg.model.data.train.dataset_path,
            uf90_output_dir=cfg.model.data.train.uf90.uniref90_path,
            sort_fastas=cfg.model.data.train.sort_fastas,
            mode="train",
            num_preprocess_workers=cfg.model.data.preprocessing.num_preprocess_workers,
        )
        # Make sure the dataset was created.
        if not os.path.isdir(cfg.model.data.train.dataset_path):
            raise ValueError(
                "Attempted to create a dataset output directory: {cfg.model.data.train.dataset_path} but it failed and was not created."
            )
        # Check input arguments for val run.
        if not os.path.exists(cfg.model.data.val.uf50_datapath):
            raise FileNotFoundError(
                "input argument ++cfg.model.data.val.uf50_datapath: {cfg.model.data.val.uf50_datapath} is not found."
            )
        preprocessor.prepare_dataset(
            uf50_datapath=cfg.model.data.val.uf50_datapath,
            uf50_output_dir=cfg.model.data.val.dataset_path,
            sort_fastas=False,
            mode="val",
        )
        # Make sure the dataset was created.
        if not os.path.isdir(cfg.model.data.val.dataset_path):
            raise ValueError(
                "Attempted to create a dataset output directory: {cfg.model.data.val.dataset_path} but it failed and was not created."
            )

        # Check input arguments for test.
        if not os.path.exists(cfg.model.data.test.uf50_datapath):
            raise FileNotFoundError(
                "input argument ++cfg.model.data.test.uf50_datapath: {cfg.model.data.test.uf50_datapath} is not found."
            )

        preprocessor.prepare_dataset(
            uf50_datapath=cfg.model.data.test.uf50_datapath,
            uf50_output_dir=cfg.model.data.test.dataset_path,
            sort_fastas=False,
            mode="test",
        )
        # Make sure the dataset was created.
        if not os.path.isdir(cfg.model.data.test.dataset_path):
            raise ValueError(
                "Attempted to create a dataset output directory: {cfg.model.data.test.dataset_path} but it failed and was not created."
            )
        
        if cfg.wandb_artifacts.wandb_use_artifacts:
            artifact = wandb.Artifact(
                name=cfg.wandb_artifacts.wandb_log_artifact_name,
                type="dataset",
                description="uniref202104_esm2_qc_test200_val200 from BioNeMo examples",
                metadata={"path":"/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200.zip"},
            )
            print(cfg.model.data.dataset_path)
            artifact.add_dir(cfg.model.data.dataset_path)
            run.log_artifact(artifact)
        

        # Downloading and preprocessing data for downstream task validation
        if cfg.model.dwnstr_task_validation.enabled:
            flip_preprocessor = FLIPPreprocess()
            if "task_name" not in cfg.model.dwnstr_task_validation.dataset:
                task_name = cfg.model.dwnstr_task_validation.dataset.dataset_path.split("/")[-1]
            else:
                task_name = cfg.model.dwnstr_task_validation.dataset.task_name
            flip_preprocessor.prepare_dataset(
                output_dir=cfg.model.dwnstr_task_validation.dataset.dataset_path, task_name=task_name
            )
        if cfg.exp_manager.create_wandb_logger:
            run.finish()

if __name__ == '__main__':
    main()
