# Import all the dependencies
import ctranslate2
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

# change this path to where you have stored your model
model_path = '/mnt/artifacts/llama2/llama2-ct'
model_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the ctranslate model
generator = ctranslate2.Generator(model_path, device=model_device)
tokenizer = transformers.AutoTokenizer.from_pretrained('subirmansukhani/llama-2-7b-miniguanaco')



prompt_template = f"<s>[INST] {{dialogue}} [/INST]"
     
#Generate the output from the LLM
def generate(prompt: str = None, pct_new_tokens: float = 1.2):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Construct the prompt for the model
    user_input = prompt_template.format(dialogue=prompt)
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(user_input))
    input_length = len(tokens)
    # new_tokens = round(pct_new_tokens*input_length)
    new_tokens = 750
    tokens_per_sec = 0
    start_time = time.time()
    results = generator.generate_batch([tokens], sampling_topk=10, max_length=new_tokens, include_prompt_in_result=False)
    end_time = time.time()
    output_text = tokenizer.decode(results[0].sequences_ids[0])
    tokens_per_sec = round(new_tokens / (end_time - start_time),3)
    
    # return {'text_from_llm': output_text, 'tokens_per_sec': tokens_per_sec}
    return {'text_from_llm': output_text}
    
