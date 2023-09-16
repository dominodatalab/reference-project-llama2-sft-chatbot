# Import all the dependencies
     
import nvidia
import os
import time
import torch
import transformers
     
from random import randint
from transformers import AutoTokenizer
     
cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir
     
# Load the ctranslate model
model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/mnt/artifacts/llama2/final_merged_checkpoint/'

generator = AutoModelForCausalLM.from_pretrained(model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    cache_dir="/mnt/artifacts/llama2-model-cache/",
    torch_dtype=torch.float16,
    device_map=device_map,
)
# load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

prompt_template = f"<s>[INST] {{prompt}} [/INST]"
     
#Generate the output from the LLM
def generate(prompt: str = None, new_tokens: int = 200):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Construct the prompt for the model
    prompt = prompt_template.format(dialogue=prompt)
    # Tokenize the prompt
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
    max_length = len(tokens) + new_tokens
    tokens_per_sec = 0
    start_time = time.time()
    results = generator.generate_batch([tokens], sampling_topk=10, max_length=new_tokens, include_prompt_in_result=False)
    end_time = time.time()
    output_text = tokenizer.decode(results[0].sequences_ids[0])
    tokens_per_sec = round(new_tokens / (end_time - start_time),3)
    return {'text_from_llm': output_text, 'tokens_per_sec': tokens_per_sec}