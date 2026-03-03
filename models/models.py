import torch
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from .gpt import call_gpt
generation_config = GenerationConfig(
    do_sample=True,
    temperature=1,
    num_beams=5,
)

def init_model(modelName, device):
    if 'gpt' in modelName:
        return None, modelName
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = LlamaForCausalLM.from_pretrained(modelName, device_map=device)
    return tokenizer, model

def get_answer(prompt, tokenizer, model, device, modelType=False, user_content="", system=False, system_content=""):
    if modelType:
        output = call_gpt(model=model, user_content=user_content, system=system, system_content=system_content)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        inputs = tokenizer(prompt, return_tensors="pt")
        generation_output = model.generate(
            input_ids=inputs.input_ids.to(device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=256,
            attention_mask=torch.ones(input_ids.shape, dtype=torch.long, device=device),
            pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(generation_output.sequences[0])
        output = output.split('<|end_header_id|>')[-1]
    return output

def get_probability(input_text, tokenizer, model, device, ans):

    res = []
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device))
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    for ans_token in ans:
        token_id = tokenizer.convert_tokens_to_ids(ans_token)
        token_prob = probs[0][-1][token_id].cpu().numpy().item()
        res.append(token_prob)
    print('res:', res)
    return res