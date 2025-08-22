# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray
import torch

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    bin_flag = False # to adapt to binary classification task.
    if config.rollout.response_length == 1:
        bin_flag = True
    traj_flag = False
    if config.rollout.calculate_log_probs:
        traj_flag = True
    if not config.data.get('data.enable_qwen3_thinking', 'True'):
        think_flag = False

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    if bin_flag:
        new_chat_lst = []
        for chat in chat_lst:
            if chat[-1]["role"] == "user":
                new_chat = chat.copy()
                new_chat.append({"role": "assistant", "content": "<judgement>"})
                new_chat_lst.append(new_chat)
            else:
                new_chat_lst.append(chat)
        chat_lst = new_chat_lst
        yes_token_id = tokenizer.encode(" YES", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode(" NO", add_special_tokens=False)[0]
        print(f"' YES' token id encoded: {tokenizer.encode(' YES', add_special_tokens=False)}")
        print(f"'YES' token id encoded: {tokenizer.encode('YES', add_special_tokens=False)}")
        print(f"' NO' token id encoded: {tokenizer.encode(' NO', add_special_tokens=False)}")
        print(f"'NO' token id encoded: {tokenizer.encode('NO', add_special_tokens=False)}")
        # assert tokenizer.encode(" YES", add_special_tokens=False) == tokenizer.encode("YES", add_special_tokens=False), f'Inconsistent encoding for "YES": {tokenizer.encode(" YES", add_special_tokens=False)} vs {tokenizer.encode("YES", add_special_tokens=False)}'
        # assert tokenizer.encode(' NO', add_special_tokens=False) == tokenizer.encode('NO', add_special_tokens=False), f'Inconsistent encoding for "NO": {tokenizer.encode(" NO", add_special_tokens=False)} vs {tokenizer.encode("NO", add_special_tokens=False)}'
    if not think_flag:
        new_chat_lst = []
        for chat in chat_lst:
            if chat[-1]["role"] == "user":
                new_chat = chat.copy()
                new_chat.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
                new_chat_lst.append(new_chat)
            else:
                new_chat_lst.append(chat)
        chat_lst = new_chat_lst
        print(f"Example of last chat: {chat_lst[0]}")
        print(f"""Example of ids: {str(tokenizer.apply_chat_template(
            chat_lst[0],
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ))}""")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    entropy_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            batch_entropies = []
            trajs_logprobs = output.batch.get("trajs_logprobs", None)
            first_logprobs = output.non_tensor_batch.get("first_logprobs", None)
            for i in range(len(output)):
                data_item = output[i]
                if traj_flag:
                    logprobs = trajs_logprobs[i]
                if bin_flag:
                    logit_dict = first_logprobs[i]
                    print(logit_dict)
                    yes_logit = logit_dict[yes_token_id]
                    no_logit = logit_dict[no_token_id]
                    if yes_logit == -float('inf') and no_logit == -float('inf'):
                        response_str = "<judgement>UNK</judgement>"
                    elif yes_logit >= no_logit:
                        response_str = "<judgement>YES</judgement>"
                    elif yes_logit < no_logit:
                        response_str = "<judgement>NO</judgement>"
                else:
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    if traj_flag:
                        actual_logprobs = logprobs[:valid_response_length]
                        if valid_response_length > 0:
                            entropy_sum = 0.0
                            for token_logprob in actual_logprobs:
                                prob = np.exp(token_logprob)
                                entropy_sum += -prob * token_logprob
                            avg_entropy = entropy_sum / valid_response_length
                        else:
                            avg_entropy = 0.0
                        batch_entropies.append(avg_entropy)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)
            entropy_lst[n_sample].extend(batch_entropies)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    entropy_lst = np.array(entropy_lst, dtype=float)
    entropy_lst = np.transpose(entropy_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    if traj_flag:
        dataset["entropy"] = entropy_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
