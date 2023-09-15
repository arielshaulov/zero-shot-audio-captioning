import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(model,
             tokenizer,
             instruction,
             input=None,
             temperature=0.1,
             top_p=0.75,
             top_k=40,
             num_beams=1,
             max_new_tokens=128,
             **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=900 + 1)
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def load_all(device, LORA_WEIGHTS = "lora-weights"):
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    LOAD_8BIT = True
    BASE_MODEL = "decapoda-research/llama-7b-hf"

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

    if not LOAD_8BIT:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer

if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm

    device = "cuda"
    model, tokenizer = load_all(device)
    content = 'Car honk'
    res = evaluate(model, tokenizer, instruction='You will assesses the audibility of a given sentence and assigns a score between 0 and 100. You should consider factors such as the coherence of words, grammatical correctness, and the likelihood that the sentence represents a meaningful auditory scenario', input=content, num_beams=4, max_new_tokens=128)
    print(res)
