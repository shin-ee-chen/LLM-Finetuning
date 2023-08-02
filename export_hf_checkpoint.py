import os
import argparse

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

# BASE_MODEL = os.environ.get("BASE_MODEL", None)
# assert (
#     BASE_MODEL
# ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

parser = argparse.ArgumentParser()

parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", type=str,
                        help='Specify path to base model')

# parser.add_argument("--adapter_model", default="/home/xchen/LLM-Finetuning/lora-alpaca", type=str,
#                         help='Specify path to adapter model')


parser.add_argument("--adapter_model", default="tloen/alpaca-lora-7b", type=str,
                        help='Specify path to adapter model')

args = parser.parse_args()


BASE_MODEL = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

# lora_model = PeftModel.from_pretrained(
#     base_model,
#     "tloen/alpaca-lora-7b",
#     device_map={"": "cpu"},
#     torch_dtype=torch.float16,
# )

lora_model = PeftModel.from_pretrained(
    base_model,
    args.adapter_model,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

model_name = args.adapter_model.split("/")[-1]
LlamaForCausalLM.save_pretrained(
    base_model, f"./{model_name}_hf", state_dict=deloreanized_sd, max_shard_size="400MB"
)

# Save tokenizer for HELM evaluation 
tokenizer.save_pretrained(f"./{model_name}_hf")

print("Finish")