
import argparse

import pytorch_lightning as pl
import torch

from datamodule import DATA_MODULE_MAP
from module import get_module


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="rte")
parser.add_argument("--t5_model", default="t5-large")
parser.add_argument("--checkpoint_path", default="/")

parser.add_argument("--prompt_tuning", action="store_true") # , default=True
parser.add_argument("--prompt_length", type=int, default=30)
parser.add_argument("--prompt_scaling", action="store_true") # , default=True
parser.add_argument("--element_wise_scaling", action="store_true")
parser.add_argument("--peft", action="store_true", default=True, help="Train only added parameters")

parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

torch.set_float32_matmul_precision("medium")

data_module = DATA_MODULE_MAP[args.task](model_name_or_path=args.t5_model,
                                        tokenizers_parallelism=True,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size)

model_class, module_kwargs = get_module(args, module_kwargs)
model = model_class(**module_kwargs,
                    dataset_name=data_module.DATASET,
                    task_name=data_module.TASK,
                    tokenizer=data_module.tokenizer,
                    inverse_label_mapping=inverse_label_mapping,
                    num_labels=len(data_module.LABELS),
                    post_processing_fn=data_module.index_processing if args.task=="multirc" else None
                    )
model.cuda(args.gpu_id)

trainer = pl.Trainer(
        enable_progress_bar=True,
        precision=args.precision,
        accelerator="auto",
        devices=1,
    )
trainer.fit(model, datamodule=data_module)