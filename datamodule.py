
import os
from typing import Any

import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import BatchEncoding


class DataModule(pl.LightningDataModule):
    """A ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face datasets."""

    DATASET = None
    TASK = None
    FIELD_PREFIX_MAP = {}
    LABELS = ()
    LOADER_COLUMNS = (
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
        "decoder_input_ids",
    )

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        tokenizers_parallelism: bool = True,
        **dataloader_kwargs: Any,
    ):
        r"""Initialize the ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            model_name_or_path (str):
                Can be either:
                    - A string, the ``model id`` of a pretrained model hosted inside a model repo on huggingface.co.
                        Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                        a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a ``directory`` containing model weights saved using
                        :meth:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
            max_seq_length (int, optional): Length to which we will pad sequences or truncate input. Defaults to 128.
            train_batch_size (int, optional): Training batch size. Defaults to 16.
            eval_batch_size (int, optional): Batch size to use for validation and testing splits. Defaults to 16.
            tokenizers_parallelism (bool, optional): Whether to use parallelism in the tokenizer. Defaults to True.
            \**dataloader_kwargs: Arguments passed when initializing the dataloader
        """
        super().__init__()
        assert self.TASK, "Make class for an appropriate task."
        assert self.DATASET, "Set dataset name"
        self.field_prefix_map = self.FIELD_PREFIX_MAP
        self.label_text_from_id = self.LABELS
        self.dataloader_kwargs = {
            "num_workers": dataloader_kwargs.get("num_workers", 4),
            "pin_memory": dataloader_kwargs.get("pin_memory", False),
        }
        self.save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.hparams.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path, use_fast=True, local_files_only=False
        )
        ...

    def setup(self, stage):
        """Setup our dataset splits for training/validation."""
        self.dataset = datasets.load_dataset(self.DATASET, self.TASK)
        print(len(self.dataset["train"]))

        if len(self.dataset["train"]) > 10000:
            self.dataset["train"] = datasets.load_dataset(self.DATASET, self.TASK, split="train[:-1000]")
            self.dataset["validation"] = datasets.load_dataset(self.DATASET, self.TASK, split="train[-1000:]")
            self.dataset["test"] = datasets.load_dataset(self.DATASET, self.TASK, split="validation")
        else:
            self.dataset["validation"] = datasets.load_dataset(self.DATASET, self.TASK, split="validation[:50%]")
            self.dataset["test"] = datasets.load_dataset(self.DATASET, self.TASK, split="validation[50%:]")
        print("len train data: ", len(self.dataset["train"]))
        print("len valid data: ", len(self.dataset["validation"]))
        print("len test data: ", len(self.dataset["test"]))

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)


    def train_dataloader(self):
        return DataLoader(self.dataset["train"],
                          batch_size=self.hparams.train_batch_size,
                          **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"],
                          batch_size=self.hparams.eval_batch_size,
                          **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"],
                          batch_size=self.hparams.eval_batch_size,
                          **self.dataloader_kwargs)

    def _get_text_pairs(self, example_batch, fields):
        raise NotImplementedError

    def _convert_to_features(self, example_batch) -> BatchEncoding:
        """Convert raw text examples to a :class:`~transformers.tokenization_utils_base.BatchEncoding` container
        (derived from python dict) of features that includes helpful methods for translating between word/character
        space and token space.

        Args:
            example_batch ([type]): The set of examples to convert to token space.

        Returns:
            ``BatchEncoding``: A batch of encoded examples (note default tokenizer batch_size=1000)
        """
        fields = list(self.field_prefix_map.keys())
        # add prefix like 'sentence1: '
        for field in fields:
            example_batch[field] = [self.field_prefix_map[field] + text for text in example_batch[field]]

        text_pairs = self._get_text_pairs(example_batch, fields)
        text = [" ".join(text) for text in text_pairs]
        features = self.tokenizer.batch_encode_plus(
            text, max_length=self.hparams.max_seq_length,
            padding="longest", truncation=True, return_tensors="pt",
        )

        # Convert label string into token ids.
        features["labels"] = self.tokenizer.batch_encode_plus(
                                [self.label_text_from_id[int_label] for int_label in example_batch["label"]],
                                max_length=self.hparams.max_seq_length,
                                padding=True, return_tensors="pt", truncation=True,
                            )['input_ids']

        num_examples = len(example_batch['idx'])
        features["decoder_input_ids"] = torch.Tensor([[0]] * num_examples).int()

        return features


class GlueDataModule(DataModule):
    DATASET = "glue"


class SuperGlueDataModule(DataModule):
    DATASET = "super_glue"


##############################################
# Glue dataset module
class SST2DataModule(GlueDataModule):
    TASK = "sst2"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class MRPCDataModule(GlueDataModule):
    TASK = "mrpc"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class CoLADataModule(GlueDataModule):
    TASK = "cola"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class RTEDataModule(GlueDataModule):
    TASK = "rte"
    FIELD_PREFIX_MAP = {
        "sentence1": "rte sentence1: ",
        "sentence2": "sentence2: ",
    }
    # for SuperGlue
    # FIELD_PREFIX_MAP = {
    #     "hypothesis": "rte sentence1: ",
    #     "premise": "sentence2: ",
    # }
    LABELS = ("entailment", "not_entailment")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]]))


class QNLIDataModule(GlueDataModule):
    TASK = "qnli"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class STSBDataModule(GlueDataModule):
    TASK = "stsb"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class MNLIDataModule(GlueDataModule):
    TASK = "mnli"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


class QQPDataModule(GlueDataModule):
    TASK = "qqp"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


################################################
# SuperGlue dataset module
# Not in T5 paper
class AxbDataModule(SuperGlueDataModule):
    TASK = "axb"
    FIELD_PREFIX_MAP = {
        "sentence1": "axb sentence1: ",
        "sentence2": "sentence2: "
    }
    LABELS = ("entailment", "not_entailment")


# Not in T5 paper
class AxgDataModule(SuperGlueDataModule):
    TASK = "axg"
    FIELD_PREFIX_MAP = {
        "hypothesis": "axg hypothesis: ",
        "premise": "premise: ",
    }
    LABELS = ("entailment", "not_entailment")


class BoolQDataModule(SuperGlueDataModule):
    TASK = "boolq"
    FIELD_PREFIX_MAP = {
        "question": "boolq qeustion: ",
        "passage": "passage: ",
    }
    LABELS = ("False", "True")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]]))

# RTE on Glue DataModule list.


class CBDataModule(SuperGlueDataModule):
    TASK = "cb"
    FIELD_PREFIX_MAP = {
        "hypothesis": "cb hypothesis: ",
        "premise": "premise: "
    }
    LABELS = ("entailment", "contradiction", "neutral")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]]))


class COPADataModule(SuperGlueDataModule):
    TASK = "copa"
    FIELD_PREFIX_MAP = {
        "choice1": "copa choice1: ",
        "choice2": "choice2: ",
        "premise": "premise: ",
        "question": "question: "
    }
    LABELS = ("False", "True")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]], example_batch[fields[2], example_batch[fields[3]]]))


class MultiRCDataModule(SuperGlueDataModule):
    TASK = "multirc"
    FIELD_PREFIX_MAP = {
        "question": "multirc question: ",
        "answer": "answer: ",
        "paragraph": "paragraph: ",
    }
    LABELS = ("False", "True")
    LOADER_COLUMNS = (
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
        "paragraph",
        "question",
        "answer",
    )

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]], example_batch[fields[2]]))

    def _convert_to_features(self, example_batch) -> BatchEncoding:
        features = super()._convert_to_features(example_batch)

        features["paragraph"] = [idx["paragraph"] for idx in example_batch["idx"]]
        features["question"] = [idx["question"] for idx in example_batch["idx"]]
        features["answer"] = [idx["answer"] for idx in example_batch["idx"]]
        return features

    def index_processing(self, preds_ids, labels_ids, batch):
        # When generating True/False is falied, it should be regarded as wrong
        preds_ids = [pred_id if pred_id != -1 else (labels_ids[i] + 1) % 2 for i, pred_id in enumerate(preds_ids)]

        # postprocessing for evaluate metric index
        paragraph_list = batch["paragraph"].detach().cpu().numpy().tolist()
        question_list = batch["question"].detach().cpu().numpy().tolist()
        answer_list = batch["answer"].detach().cpu().numpy().tolist()
        preds_ids = [{"idx": {"answer": ans, "paragraph": para, "question": q}, "prediction": pred}
                     for ans, para, q, pred in zip(answer_list, paragraph_list, question_list, preds_ids)]
        return preds_ids


# 복잡, Not in XPrompt
class ReCoRDDataModule(SuperGlueDataModule):
    TASK = "record"
    FIELD_PREFIX_MAP = {
    }
    LABELS = ("entailment", "not_entailment")


class WiCDataModule(SuperGlueDataModule):
    """
    https://github.com/google-research/text-to-text-transfer-transformer/blob/ba171b6f94eafcee60d0714fd6d60749b572d1f2/t5/data/tasks.py#L77
    # This ignores the start/end indices which show where in each sentence the
    # word appears.
    # TODO(craffel): Investigate using those indices.
    """
    TASK = "wic"
    FIELD_PREFIX_MAP = {
        # "???": "wic pos: ",
        "sentence1": "sentence1: ",
        "sentence2": "sentence2: ",
        "word": "word: ",
    }
    LABELS = ("False", "True")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]]))


class WSCDataModule(SuperGlueDataModule):
    TASK = "wsc.fixed"
    FIELDS = ["text", "span1_index", "span2_index", "span1_text", "span2_text", "label"]
    LABELS = ("False", "True")

    def _get_text_pairs(self, example_batch, fields):
        return list(zip(example_batch[fields[0]], example_batch[fields[1]],
                        example_batch[fields[2]], example_batch[fields[3]],
                        example_batch[fields[4]], example_batch[fields[5]]))

    def _convert_to_features(self, example_batch) -> BatchEncoding:
        records = self._get_text_pairs(example_batch, self.FIELDS)

        text = []
        for record in records:
            tokens = record[0].split()
            span1_text_num_tokens = len(record[3].split())
            indices = list(range(record[1], record[1] + span1_text_num_tokens))
            tokens[indices[0]] = "*" + tokens[indices[0]]
            tokens[indices[-1]] = tokens[indices[-1]] + "*"
            tokens[record[2]] = "#" + tokens[record[2]] + "#"
            processed_text = "wsc: " + " ".join(tokens)
            text.append(processed_text)

        features = self.tokenizer.batch_encode_plus(
            text, max_length=self.hparams.max_seq_length,
            padding="longest", truncation=True, return_tensors="pt",
        )

        # Convert label string into token ids.
        features["labels"] = self.tokenizer.batch_encode_plus(
                                [self.label_text_from_id[int_label] for int_label in example_batch["label"]],
                                max_length=self.hparams.max_seq_length,
                                padding=True, return_tensors="pt", truncation=True,
                            )['input_ids']

        return features


DATA_MODULE_MAP = {
    "sst2": SST2DataModule,
    "mrpc": MRPCDataModule,
    "cola": CoLADataModule,
    "rte": RTEDataModule,
    "qnli": QNLIDataModule,
    "stsb": STSBDataModule,
    "mnli": MNLIDataModule,
    "qqp": QQPDataModule,
    "axb": AxbDataModule,
    "axg": AxgDataModule,
    "boolq": BoolQDataModule,
    "cb": CBDataModule,
    "copa": COPADataModule,
    "multirc": MultiRCDataModule,
    "record": ReCoRDDataModule,
    "wic": WiCDataModule,
    "wsc": WSCDataModule,
}
