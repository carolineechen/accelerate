# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
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
import argparse

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from accelerate.utils import PrepareForLaunch, patch_environment

import runhouse as rh


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# on remote self hosted hardware. This can be either
# - an on demand cluster (AWS, GCP, Azure, Lambda)
# - an existing cluster you already have up, using SSH credentials
# 
# To run this script, you will need to `pip install runhouse`,
# and set up cloud credentials following `sky check` instructions
# to use an on-demand cluster, or if you are using an existing cluster,
# you will need to modify the args when calling get_existing_cluster().
# > python self_hosted_compute.py
#
# New additions from the base script can be found quickly by
# looking for the # New Code # tags
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


# New Code #
def get_on_demand_cluster(
        cluster_name='rh-cluster',
        instance_type='V100:4',
        provider='cheapest',
        autostop_mins=30,
    ):
    """
    Instantiates an on-demand cluster using Runhouse, to launch distributed training on.

    Args:
        cluster_name (str):
            Name to assign the cluster
        instance_type (str):
            Type and number of cloud instance to use for the cluster. ex/ A100:1, V100:4, CPU:4+
        provider (str):
            Cloud provider to use, or 'cheapest' will choose the cheapest enabled provider for the
            requested compute. Options: ['aws', 'azure', 'gcp', 'lambda', 'cheapest']
        autostop_mins (int):
            Cluster will automatically terminate after this many minutes of inactivity, or it will
            stay up indefinitely if `-1` is passed in.
    """
    return rh.cluster(
        name=cluster_name,
        instance_type=instance_type,
        provider=provider,
        use_spot=False,
        autostop_mins=autostop_mins,
    )

# New Code #
def get_existing_cluster(
        ips: list,
        ssh_creds: dict,
        cluster_name='rh-cluster'
):
    """
    Instantiates a remote existing cluster using Runhouse, to launch distributed training on.
    
    Args:
        ips (List[str]):
            List of IP address(es) for the cluster. ['000.00.00.000', '....']
        ssh_creds (Dict):
            Dictionary of ssh_creds {'user': <user>, 'ssh_private_key': <path_to_private_key>}
        cluster_name (str):
            Name to assign the cluster
    """
    return rh.cluster(
        name=cluster_name,
        ips=ips,
        ssh_creds=ssh_creds,
    )

# New Code #
def launch_training(training_function, *args):
    num_processes = torch.cuda.device_count()
    print(f'Device count: {num_processes}')
    with patch_environment(world_size=num_processes, master_addr="127.0.01", master_port="29500",
                           mixed_precision=args[1].mixed_precision):
        launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
        torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")



def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.
    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    
    # New Code #
    cluster = get_on_demand_cluster(cluster_name='rh-multigpu', autostop_mins=30)  # or `get_existing_cluster` function
    reqs = ['accelerate', 'transformers', 'datasets', 'evaluate','tqdm', 'scipy', 'scikit-learn', 'tensorboard',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117']
    
    cluster.up_if_not()
    cluster.restart_grpc_server()
    cluster_train = rh.function(fn=training_function).to(system=cluster, reqs=reqs)
    cluster_launch_training = rh.function(fn=launch_training).to(system=cluster)

    cluster_launch_training(cluster_train, config, args, stream_logs=True)


if __name__ == "__main__":
    main()