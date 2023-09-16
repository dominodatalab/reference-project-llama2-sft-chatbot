# Import all the dependencies
     
import nvidia
import os
import time
import torch
import transformers
     
from random import randint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig 
     
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
    device_map='auto',
)
# load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

prompt_template = f"<s>[INST] {{dialogue}} [/INST]"
     
#Generate the output from the LLM
def generate(prompt: str = None, new_tokens: int = 200):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Construct the prompt for the model
    user_input = prompt_template.format(dialogue=prompt)
    
    tokens_per_sec = 0
    start_time = time.perf_counter()
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids
    input_ids = input_ids.to('cuda')

    generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens = new_tokens
        )

    with torch.no_grad():
        generated_ids = generator.generate(input_ids, generation_config=generation_config)
    
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end_time = time.perf_counter()
    gen_text = gen_text.replace(f"[INST] {prompt} [/INST]", '')
    tokens_per_sec = round(new_tokens / (end_time - start_time),3)
    return {'text_from_llm': gen_text, 'tokens_per_sec': tokens_per_sec}
    
