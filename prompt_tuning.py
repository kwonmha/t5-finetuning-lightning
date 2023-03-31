
import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datamodule import DATA_MODULE_MAP
from module import get_module
from utils import get_result_path

METRIC_MAP = {
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "cola": "matthews_correlation",
    "rte": "accuracy",
    "boolq": "accuracy",
    "cb": "accuracy",
    "copa": "accuracy",
    "multirc": "exact_match",
    "wic": "accuracy",
    "wsc": "accuracy"
}


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="rte")
parser.add_argument("--t5_model", default="t5-base")

parser.add_argument("--prompt_tuning", action="store_true") # , default=True
parser.add_argument("--prompt_length", type=int, default=30)
parser.add_argument("--prompt_scaling", action="store_true") # , default=True
parser.add_argument("--element_wise_scaling", action="store_true")
# parser.add_argument("--decoder_prompt", action="store_true") # , default=True 
parser.add_argument("--peft", action="store_true", default=True, help="Train only added parameters")

parser.add_argument("--lr", type=float, default=3e-4) # 그냥 finetuning할 때는 얼마인지?
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--train_batch_size", type=int, default=4) # 3b, Full finetuning - 8에서 X
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=50) # T1.1, 30000 steps, bs 32, lr 0.3 in Lester et al.

parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

torch.set_float32_matmul_precision("medium")

result_path = get_result_path(args)

pl.seed_everything(args.seed)
data_module = DATA_MODULE_MAP[args.task](model_name_or_path=args.t5_model, 
                                        tokenizers_parallelism=True,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size)
inverse_label_mapping = {label: i for i, label in enumerate(data_module.LABELS)}

optimizer_init = {"weight_decay": 1e-05, "eps": 1e-07, "lr": args.lr}
# lr_scheduler_init = {"T_0": 1, "T_mult": 2, "eta_min": 1e-07}
module_kwargs = {
    "model_name_or_path": args.t5_model,
    "optimizer_init": optimizer_init,
    # "lr_scheduler_init": lr_scheduler_init,
}

model_class, module_kwargs = get_module(args, module_kwargs)
model = model_class(**module_kwargs,
                    dataset_name=data_module.DATASET,
                    task_name=data_module.TASK,
                    tokenizer=data_module.tokenizer,
                    inverse_label_mapping=inverse_label_mapping,
                    num_labels=len(data_module.LABELS),
                    post_processing_fn=data_module.index_processing if args.task == "multirc" else None
                    )
model.cuda(args.gpu_id)

early_stop_callback = EarlyStopping(monitor=METRIC_MAP[args.task], min_delta=0.0001, patience=5, verbose=False, mode="max")
logger = CSVLogger(result_path, name=args.task)

checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                      dirpath=logger.log_dir,
                                      # every_n_train_steps=1000,
                                      monitor=METRIC_MAP[args.task],
                                      mode="max")

trainer = pl.Trainer(
    enable_progress_bar=True,
    max_epochs=args.max_epochs,
    # max_steps=262144,
    precision=args.precision, # important!!! KT 서버에서 그냥 16으로 하면 안됨. 학교 서버는 모름
    accelerator="auto",
    devices=1,
    logger=logger,
    accumulate_grad_batches=args.gradient_accumulation_steps,
    callbacks=[early_stop_callback, checkpoint_callback],
)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module, ckpt_path="best")


