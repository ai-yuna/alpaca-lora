import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

'''
伪代码：
1. 分布式训练设置device_map
2. wandb设置
3. 加载模型 model = LlamaForCausalLM.from_pretrained(
4. 加载分词器 LlamaTokenizer.from_pretrained(base_model) 设置pad_token_id padding_side
5. lora配置，并且构建合并peft model：get_peft_model(lora_config,model)
6. 加载数据集，load_dataset，如果设置val数据集，那就切分数据集train_test_split，
    然后执行拼接数据为完整的prompt，并tokenize（如果不在instruction+input上进行训练，那就更新lables中这些位置的id为-100，不参与训练）
7. 创建transformers.trainer()并配置参数
8. 重新定义模型的state_dict方法，get_peft_model_state_dict(self, old_state_dict())
9. 编译模型，模型训练，模型保存: model = torch.compile(model),trainer.train(),model.save_pretrained(output_dir)
'''
def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,  # 每块gpu上的实际batch_size
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,  # maxseqlen
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"  # 检查 base_model 是否有值（即它不是空字符串

    gradient_accumulation_steps = batch_size // micro_batch_size  # 梯度累计

    prompter = Prompter(prompt_template_name)

    # 分布式训练设置
    device_map = "auto"  # 自动选择可用的计算设备（例如GPU）
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 分布式训练中的总进程数
    ddp = world_size != 1
    # 如果正在进行分布式训练
    if ddp:
        device_map = {"": int(os.environ.get(
            "LOCAL_RANK") or 0)}  ## 获取当前进程的 LOCAL_RANK，并将其设置为设备映射，这确保每个进程使用一个独立的计算设备，值是从 "LOCAL_RANK" 环境变量获取的整数值，或默认值 0
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # 判断是否使用wandb
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    # 设置对应的环境变量
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # 加载模型
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,  # 使用8位精度可以减少模型的内存使用量并提高推理速度，但可能会牺牲一些精度。
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # 加载分词器
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (  # 设置填充令牌（pad token）的ID
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    ## 分词
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        # tokenize
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(
                1)  # 值 1 表示模型应该关注该位置的令牌。在进行自注意力计算时，这个令牌会被考虑进去。值 0 表示模型应该忽略该位置的令牌。这通常用于填充令牌或其他模型不应该关注的令牌。

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point): #输出原始数据，拼接完整的prompt，并分词
        ## 拼接生成完整的prompt str，基本包含指令，输入，输出
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:  # 不在instruction+input上进行训练
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            # instruction+input部分的标签被设置为 - 100（这是一个特殊值，表示在计算损失时应忽略这些令牌），而其余部分使用完整提示的标签。
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    ## lora配置及lora model
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)  # 合并model lora

    # 加载数据
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # 从检查点继续（或恢复）训练，并据此加载模型权重
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin" #LoRA模型的检查点文件
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name) #加载检查点文件中的权重
            set_peft_model_state_dict(model, adapters_weights)  # 设置模型的状态字典。状态字典是一个包含模型所有参数（如权重和偏置）的Python字典
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    ## 如果有val数据集，就切分数据集，并拼接完整的prompt
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    #没有设置分布式训练，但是有多块gpu
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False


    old_state_dict = model.state_dict
    # 重新定义模型的state_dict方法
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model)) #`.__get__(model, type(model))` 是一个技巧，用于将这个 lambda 函数绑定到模型实例上，使其能像原始的方法一样被调用

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
