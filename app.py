import requests
import streamlit as st

import nvidia
import os
import time
import torch
import transformers

from random import randint
from streamlit.web.server import websocket_headers
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


#Generate the output from the LLM
def generate(prompt: str = None, new_tokens: int = 200):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Construct the prompt for the model
    user_input = prompt_template.format(dialogue=prompt)
    
    tokens_per_sec = 0
    start_time = time.perf_counter()
    gen_text = pipe_llama7b_chat(user_input)
    
#     input_ids = tokenizer(user_input, return_tensors="pt").input_ids
#     input_ids = input_ids.to('cuda')

#     generation_config = GenerationConfig(
#             pad_token_id=tokenizer.pad_token_id,
#             max_new_tokens = new_tokens
#         )

#     with torch.no_grad():
#         generated_ids = generator.generate(input_ids, generation_config=generation_config)
    
#     gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     end_time = time.perf_counter()
#     gen_text = gen_text.replace(f"[INST] {prompt} [/INST]", '')

    tokens_per_sec = round(new_tokens / (end_time - start_time),3)
    return {'text_from_llm': gen_text, 'tokens_per_sec': tokens_per_sec}
    
    
cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir


# Load the Huggingface model
# model_path = '/mnt/artifacts/llama2/final_merged_checkpoint/'
model_path = 'subirmansukhani/llama-2-7b-miniguanaco'
model_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the Huggingface model
# Reload model in FP16 and merge it with LoRA weights
generator = AutoModelForCausalLM.from_pretrained(model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
#     cache_dir="/mnt/artifacts/llama2-model-cache/",
    torch_dtype=torch.float16,
    device_map='auto',
)
# load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

pipe_llama7b_chat = pipeline(task="text-generation", model=generator, tokenizer=tokenizer, max_length=200, return_full_text=False, device=1) 

prompt_template = f"<s>[INST] {{dialogue}} [/INST]"


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'tokens_sec' not in st.session_state:
    st.session_state['tokens_sec'] = []


st.set_page_config(initial_sidebar_state='collapsed')
clear_button = st.sidebar.button("Clear Conversation", key="clear")
                


if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['tokens_sec'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=150)
        submit_button = st.form_submit_button(label='Send')
        tokens_sec = 0
    if submit_button and user_input :
        with st.spinner("Generating response"):
            prompt = prompt_template.format(dialogue=user_input)
            llm_response = generate(prompt)
            answer = llm_response['text_from_llm']
            tokens_sec = llm_response['tokens_per_sec']
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer)
        st.session_state['tokens_sec'].append(tokens_sec)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(f"Tokens generated per sec: {st.session_state['tokens_sec'][i]}")
