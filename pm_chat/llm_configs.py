
#from chatbot_ui import contents

# Define prompts for local LLMs
alpaca_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {contents}

    ### Response:"""

llama_prompt = f"""
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant.
    <</SYS>>
    {contents} [/INST]</s>"""

mistral_prompt = f"""<s>[INST] Please respond to the Question : {contents} [/INST]"""
