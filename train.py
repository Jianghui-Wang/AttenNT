import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    full_finetune : Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None)
    use_lora: bool = field(default=True)
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.)
    use_attent_simple: Optional[bool] = field(default=False)
    attent_ratio: Optional[float] = field(default=0.7)
    attent_selection_epoch: Optional[int] = field(default=1)
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    data_path: str = field(default=None)
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train")
    dataset_field: List[str] = field(default=None)
    shuffle_dataset : Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    merge : Optional[bool] = field(default=False)

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

class AtteNTSimpleTrainer(Trainer):
    def __init__(self, *args, attent_ratio=0.7, use_attent_simple=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.attent_ratio = attent_ratio
        self.use_attent_simple = use_attent_simple
        self.sample_losses = {}
        self.selected_indices = None
        self.is_selection_phase = True
        self.original_train_dataset = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.get("loss")
        
        if self.use_attent_simple and self.is_selection_phase and self.state.epoch < 1.0:
            if "labels" in inputs:
                labels = inputs["labels"]
                logits = outputs.get("logits")
                
                if logits is not None and labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    batch_size = shift_logits.shape[0]
                    vocab_size = shift_logits.shape[-1]
                    
                    shift_logits_flat = shift_logits.view(-1, vocab_size)
                    shift_labels_flat = shift_labels.view(-1)
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
                    token_losses = loss_fct(shift_logits_flat, shift_labels_flat)
                    
                    token_losses = token_losses.view(batch_size, -1)
                    sample_losses = token_losses.mean(dim=1)
                    
                    if "input_ids" in inputs:
                        for i, sample_loss in enumerate(sample_losses):
                            sample_key = hash(tuple(inputs["input_ids"][i].cpu().numpy()))
                            self.sample_losses[sample_key] = sample_loss.item()
        
        return (loss, outputs) if return_outputs else loss
    
    def _inner_training_loop(self, *args, **kwargs):
        if self.use_attent_simple:
            self.original_train_dataset = self.train_dataset
            logger.info("AtteNT_simple: Starting selection phase (first epoch)...")
            self.is_selection_phase = True
            
        output = super()._inner_training_loop(*args, **kwargs)
        
        if self.use_attent_simple and self.is_selection_phase and self.state.epoch >= 1.0:
            self._select_high_loss_samples()
            self.is_selection_phase = False
            
            if self.args.num_train_epochs > 1:
                logger.info(f"AtteNT_simple: Continuing training with {len(self.train_dataset)} selected samples...")
                remaining_epochs = self.args.num_train_epochs - 1
                self.args.num_train_epochs = remaining_epochs
                self.state.epoch = 0
                output = super()._inne_
