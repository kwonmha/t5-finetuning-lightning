
from datetime import datetime
from typing import Any, Dict, List, Optional

import evaluate
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR
from torch.optim.adamw import AdamW
from transformers import AutoConfig, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.utils.import_utils import is_torch_fx_proxy


class FineTuningClassification(pl.LightningModule):
    """A ``LightningModule`` that can be used to fine-tune a foundational model on either the RTE or BoolQ
    SuperGLUE tasks using Hugging Face implementations of a given model and the `SuperGLUE Hugging Face dataset."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        inverse_label_mapping,
        optimizer_init: Dict[str, Any],
        # lr_scheduler_init: Dict[str, Any],
        model_cfg: Optional[Dict[str, Any]] = None,
        dataset_name: str = None,
        task_name: str = None,
        experiment_tag: str = "default",
        num_labels: int = 0,
        post_processing_fn = None
    ):
        """
        Args:
            model_name_or_path (str): Path to pretrained model or identifier from https://huggingface.co/models
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
        """
        super().__init__()
        self.num_labels = num_labels
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, local_files_only=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.tokenizer = tokenizer
        self.inverse_label_mapping = inverse_label_mapping
        self.init_hparams = {
            "optimizer_init": optimizer_init,
            # "lr_scheduler_init": lr_scheduler_init,
            "model_config": self.model.config,
            "model_name_or_path": model_name_or_path,
            "dataset_name": dataset_name,
            "task_name": task_name,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}",
        }
        self.save_hyperparameters(self.init_hparams)
        self.metric = evaluate.load(self.hparams.dataset_name, self.hparams.task_name, experiment_id=self.hparams.experiment_id)
        self.post_processing_fn = post_processing_fn
        self.loss_func = CrossEntropyLoss(ignore_index=-100)

        self.no_decay = ["bias", "LayerNorm.weight"]

        self.gen_kwargs = {
            "max_length": self.model.config.max_length,
            "num_beams": self.model.config.num_beams,
        }

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        input_keys = ["input_ids", "attention_mask", "labels"]
        batch_input = {key: batch[key] for key in input_keys}
        outputs = self(**batch_input)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_keys = ["input_ids", "attention_mask", "labels"] #
        batch_input = {key: batch[key] for key in input_keys}

        generated_tokens = self.model.generate(batch_input["input_ids"],
                                               attention_mask=batch_input["attention_mask"],
                                               **self.gen_kwargs)

        outputs = self(**batch_input)
        # exit()
        val_loss = outputs[0]
        self.log("val_loss", val_loss, prog_bar=True)

        labels = batch_input["labels"]
        preds_cpu = generated_tokens.detach().cpu().numpy().tolist()
        labels_cpu = labels.detach().cpu().numpy().tolist()

        preds_ids = [self.inverse_label_mapping.get(self.tokenizer.decode(ids[:-1]).rstrip("</s>""<pad>"), -1) for ids in preds_cpu]
        labels_ids = [self.inverse_label_mapping[self.tokenizer.decode(ids[:-1]).rstrip("</s>""<pad>")] for ids in labels_cpu]

        # for MultiRC dataset
        if self.post_processing_fn:
            preds_ids = self.post_processing_fn(preds_ids, labels_ids, batch)

        metric_dict = self.metric.compute(predictions=preds_ids, references=labels_ids)
        self.log_dict(metric_dict, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input_keys = ["input_ids", "attention_mask"]
        batch_input = {key: batch[key] for key in input_keys}
        generated_tokens = self.model.generate(batch_input["input_ids"],
                                               attention_mask=batch_input["attention_mask"],
                                               **self.gen_kwargs)
        labels = batch["labels"]

        preds_cpu = generated_tokens.detach().cpu().numpy().tolist()
        labels_cpu = labels.detach().cpu().numpy().tolist()

        preds_ids = [self.inverse_label_mapping.get(self.tokenizer.decode(ids[:-1]).rstrip("</s>""<pad>"), -1) for ids in preds_cpu]
        labels_ids = [self.inverse_label_mapping[self.tokenizer.decode(ids[:-1]).rstrip("</s>""<pad>")] for ids in labels_cpu]

        # for MultiRC dataset
        if self.post_processing_fn:
            preds_ids = self.post_processing_fn(preds_ids, labels_ids, batch)

        metric_dict = self.metric.compute(predictions=preds_ids, references=labels_ids)
        self.log_dict(metric_dict, prog_bar=True)

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer_init["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        # the phase 0 parameters will have been set to require gradients during setup
        # you can initialize the optimizer with a simple requires.grad filter as is often done,
        # but in this case we pass a list of parameter groups to ensure weight_decay is
        # not applied to the bias parameter (for completeness, in this case it won't make much
        # performance difference)
        # optimizer = AdamW(params=self._init_param_groups(), **self.hparams.optimizer_init)
        print("learning rate:", self.hparams.optimizer_init["lr"])
        optimizer = Adafactor(params=self._init_param_groups(),
            scale_parameter=False, 
            relative_step=False, 
            warmup_init=False,
            lr=self.hparams.optimizer_init["lr"]
        )
        scheduler = ConstantLR(optimizer, 
            factor=1.0,
            total_iters=0,
            last_epoch=-1
        )
        # scheduler = {
        #     "scheduler": scheduler,
        #     "interval": "epoch"
        # }
        return [optimizer], [scheduler]


class PromptClassification(FineTuningClassification):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        inverse_label_mapping,
        optimizer_init: Dict[str, Any],
        # lr_scheduler_init: Dict[str, Any],
        model_cfg: Optional[Dict[str, Any]] = None,
        dataset_name: str = None,
        task_name: str = None,
        experiment_tag: str = "default",
        num_labels: int = 0,
        post_processing_fn = None,
        **kwargs
        # prompt_length = 5,
        # do_prompt_scaling = False,
        # peft: bool = False
    ):
        super(PromptClassification, self).__init__(
            model_name_or_path,
            tokenizer,
            inverse_label_mapping,
            optimizer_init,
            model_cfg,
            dataset_name,
            task_name,
            experiment_tag,
            num_labels,
            post_processing_fn
        )
        self.ids_embedding = self.model.encoder.embed_tokens
        self.prompt_length = kwargs.get("prompt_length")
        self.prompt_tokens = torch.arange(self.prompt_length).long()
        self.prompt_encoder = torch.nn.Embedding(self.prompt_length,
                                                 self.model.config.hidden_size)
        self.element_wise_scaling = kwargs.get("element_wise_scaling")
        if self.element_wise_scaling:
            self.prompt_scaling_params = Parameter((torch.ones((self.prompt_length, self.model.config.d_model))))
        else:
            self.prompt_scaling_params = Parameter((torch.ones((self.prompt_length, 1))))
        self.prompt_scaling = kwargs.get("prompt_scaling")
        self.peft = kwargs.get("peft")

        if self.peft:
            for param in self.model.parameters():
                param.requires_grad = False

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def get_prompt(self, batch_size):
        prefix_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        prompts = self.prompt_encoder(prefix_tokens)
        if self.prompt_scaling:
            prompts = prompts * self.prompt_scaling_params
        return prompts

    def forward(self, **inputs):
        input_keys = ["input_ids", "attention_mask", "labels"]
        batch_size = inputs["input_ids"].shape[0]

        input_embedding = self.ids_embedding(inputs["input_ids"])
        prompts = self.get_prompt(batch_size)
        input_embedding = torch.cat((prompts, input_embedding), dim=1)

        prompt_attention_mask = torch.ones(batch_size, self.prompt_length).to(self.model.device)
        attention_mask = torch.cat((prompt_attention_mask, inputs["attention_mask"]), dim=1)
        # input_embedding = inputs["input_ids"]
        # decoder_input_ids = self._shift_right(inputs["labels"])
        return self.model(
            # input_ids=inputs["input_ids"],
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            labels=inputs["labels"],
            # decoder_input_ids=decoder_input_ids
        )

    def validation_epoch_end(self, outputs):
        print(self.lr_schedulers().get_last_lr())

def get_module(args, kwargs):
    if args.prompt_tuning:
        prompt_module_kwargs = {
            "prompt_length": args.prompt_length,
            "prompt_scaling": args.prompt_scaling,
            "peft": args.peft,
            "element_wise_scaling": args.element_wise_scaling
        }
        kwargs.update(prompt_module_kwargs)
        return PromptClassification, kwargs
    return FineTuningClassification, kwargs
