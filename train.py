import os

# os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'
# os.environ['PATH'] = '/usr/local/cuda-11.3/bin:' + os.environ['PATH']


import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule,MySpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import MyScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     print("Hello, World!")
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=8, help="Accumulate gradients.")
     
     MyScoreModel.add_argparse_args(
          parser.add_argument_group("MyScoreModel", description=MyScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     
     data_module_cls = MySpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     
     model = MyScoreModel( #ncsnpp, ouve, SpecsDataModule 정의하기
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          #backbone = "ncsnpp", sde = "ouve", data_module_cls = SpecsDataModule,
          **{
               **vars(arg_groups['MyScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )
     
     # Set up logger configuration
     
     # if args.nolog:
     if True:
          logger = TensorBoardLogger("logs", name="my_experiment")
          # logger = None
     else:
          logger = WandbLogger(project="sgmse", log_model=True, save_dir="logs", name=args.wandb_name)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=f"logs/{logger.version}", 
                                       save_last=True, filename='{epoch}-{step}-last',
                                       every_n_train_steps=50 #실은 20*256=5120번마다 저장
                                       )]
          if args.num_eval_files:
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
                    save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}',
                    every_n_train_steps=1,
                    )
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{logger.version}", 
                    save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}'
                    ,every_n_train_steps=1)
               callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp", logger=logger,
          log_every_n_steps=1, 
          num_sanity_val_steps=0,
          callbacks=callbacks,
          gpus=[ 7,8],
          #gpus=[8],
          
     )

     # Train model
     

     trainer.fit(model, ckpt_path=args.ckpt)
     # trainer.save_checkpoint("example.ckpt")
