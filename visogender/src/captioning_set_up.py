"""
This file is used to set up any captioning models that will be evaluated

Author: @smhall97

"""
import torch 
torch.cuda.empty_cache()
from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2Model, AutoProcessor, \
                          AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM,Qwen2VLForConditionalGeneration
from src.data_utils import get_image, readImageFromJpg
from resolution_bias.captioning_run.caption_input_params import caption_input_params, main_dir
device_id = caption_input_params["device_id"]
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"


def llava_setup_model_processor():
    processor = AutoProcessor.from_pretrained("models/llava-1.5-7b-hf")
    llava_model = AutoModelForPreTraining.from_pretrained("models/llava-1.5-7b-hf").to(device)

    return llava_model,processor

def qwen_setup_model_tokenizer():
        device = "cuda:2" if torch.cuda.is_available() else "cpu"
        model = Qwen2VLForConditionalGeneration.from_pretrained("models/Qwen2-VL-7B-Instruct", device_map=device)
        processor = AutoProcessor.from_pretrained("models/Qwen2-VL-7B-Instruct")
        return model, processor

        tokenizer = AutoTokenizer.from_pretrained("models/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("models/Qwen2-VL-7B-Instruct",trust_remote_code=True,fp32=True).eval()
        model.to(device)
        return model, tokenizer


def blip_setup_model_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    return blip_model, processor

def blipv2_set_up_model_processor():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipv2_model =  Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

    return blipv2_model, processor

def instruct_blipv2_set_up_model_processor():
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

    processor = InstructBlipProcessor.from_pretrained("models/instructblip-vicuna-7b")
    instruct_blipv2_model =  InstructBlipForConditionalGeneration.from_pretrained("models/instructblip-vicuna-7b").to(
        device)

    return instruct_blipv2_model, processor

def qwen_get_probabilities_his_her_their(image_url: str, idx: str, system_content:str, usr_content:str,\
         text_input: str, model, tokenizer) -> List:
    raw_image = readImageFromJpg(idx)

    # token_for_his = tokenizer('his', return_tensors="pt").to("cuda")["input_ids"][0][-1]
    # token_for_her = tokenizer('her', return_tensors="pt").to("cuda")["input_ids"][0][-1]
    # token_for_their = tokenizer('their', return_tensors="pt").to("cuda")["input_ids"][0][-1]
    token_for_his = tokenizer(text=['his'],padding=True, return_tensors="pt")["input_ids"][0][-1]
    token_for_her = tokenizer(text=['her'],padding=True, return_tensors="pt")["input_ids"][0][-1]
    token_for_their = tokenizer(text=['their'],padding=True, return_tensors="pt")["input_ids"][0][-1]
    print(token_for_his,token_for_her,token_for_their)
    # inputs = tokenizer(text = f"{system_content} USER:<image>{usr_content} ASSISTANT: {text_input}",
    #                    images = raw_image,
    #                    return_tensors="pt").to(device)


    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    text_prompt = f'''<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{usr_content}<|im_end|>\n<|im_start|>assistant\n{text_input}'''
    inputs = tokenizer(text=[text_prompt], images=[raw_image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda:2')

    # Inference: Generation of the output
    # output_ids = model.generate(**inputs, max_new_tokens=128)
    #
    # inputs = tokenizer.from_list_format([{"image":f'imgs/{idx}.jpg'},
    #                                     {"text":f"{system_content} {usr_content} Description:{text_input}"}])
    with torch.no_grad():
        outputs = model(**inputs)
        if "logits" in outputs.keys():
            logits = outputs["logits"][0, -1, :]
        else:
            logits = outputs["decoder_logits"][0, -1, :]


    # logits =  torch.softmax(logits, dim=1)
    logits_for_his = logits[token_for_his]
    logits_for_her = logits[token_for_her]
    logits_for_their = logits[token_for_their]
    print({"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()})
    return {"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()}

    answer_pre_map = dict(
        Male = 0,
        Female = 0,
        Unknown = 0,
    )
    input = tokenizer.from_list_format([{"image":f'imgs/{idx}.jpg'},
                                        {"text":f"{system_content} {usr_content} {text_input}"}])
    answer, history = model.chat(tokenizer, query=input, history=None ,max_new_tokens=512)
    print(f"***{answer}***")
    for item in answer_pre_map.keys():
        if item.lower() == answer.lower().split('.')[0]:
            answer_pre_map[item] = 1
    print({"his": answer_pre_map['Male'], "her":  answer_pre_map['Female'], "their":  answer_pre_map['Unknown']})
    return {"his": answer_pre_map['Male'], "her":  answer_pre_map['Female'], "their":  answer_pre_map['Unknown']}


def llava15_get_probabilities_his_her_their(image_url: str, idx: str, system_content:str, usr_content:str,\
                                            text_input: str, model, processor) -> List:
    """"
    Returns the logits of the key phrases for "his", "her" and "their". The order of the list is preserved, and no reordering is done

    Args:
        image_url: url to the image to be captioned
        text_input: start sentence which includes the profession
        model: captioning model
        processor: captioning model processor

    Returns:
        list(tensors): logits for "his", "her" and "their". The order is used in subsequent code as well.
    """
    # raw_image = get_image(image_url)
    raw_image = readImageFromJpg(idx)

    token_for_his = processor(text=f"<image>\nhis\n",
                              images=raw_image,
                              return_tensors="pt").to("cuda")["input_ids"][0][-2]
    token_for_her = processor(text=f"<image>\nher\n",
                              images=raw_image,
                              return_tensors="pt").to("cuda")["input_ids"][0][-2]
    token_for_their = processor(text=f"<image>\ntheir\n",
                              images=raw_image,
                              return_tensors="pt").to("cuda")["input_ids"][0][-2]
    inputs = processor(text = f"{system_content} USER:<image>{usr_content} ASSISTANT: {text_input}",
                       images = raw_image,
                       return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        if "logits" in outputs.keys():
            logits = outputs["logits"][0, -1, :]
        else:
            logits = outputs["decoder_logits"][0, -1, :]

    # logits = torch.nn.functional.log_softmax(logits,dim=0)
    logits_for_his = logits[token_for_his]
    logits_for_her = logits[token_for_her]
    logits_for_their = logits[token_for_their]
    print({"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()})
    return {"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()}

    # output = model.generate(**inputs, max_new_tokens=512)  # type: ignore
    # outnpy = output.to("cpu").numpy()
    # answer = processor.decode(outnpy[0], skip_special_tokens=True)
    # answer = answer.replace(f"{system_content} USER:<image>{usr_content} '{text_input}' ASSISTANT: ", "").strip()
    # answer = answer.split("ASSISTANT: ")[-1]
    # answer_pre_map = dict(
    #     Male = 0,
    #     Female = 0,
    #     Unknown = 0,
    # )
    # print(f"***{answer}***")
    # for item in answer_pre_map.keys():
    #     if item.lower() == answer.lower().split('.')[0]:
    #         answer_pre_map[item] = 1
    # print({"his": answer_pre_map['Male'], "her":  answer_pre_map['Female'], "their":  answer_pre_map['Unknown']})
    # return {"his": answer_pre_map['Male'], "her":  answer_pre_map['Female'], "their":  answer_pre_map['Unknown']}
    # return {"his": logits_for_male.item(), "her": logits_for_female.item(), "their": 0}

def blip_get_probabilities_his_her_their(image_url: str, idx: str, text_input: str, model, processor)->List:
    """"
    Returns the logits of the key phrases for "his", "her" and "their". The order of the list is preserved, and no reordering is done

    Args:
        image_url: url to the image to be captioned
        text_input: start sentence which includes the profession
        model: captioning model
        processor: captioning model processor

    Returns:
        list(tensors): logits for "his", "her" and "their". The order is used in subsequent code as well. 
    """
    # raw_image = get_image(image_url)
    raw_image = readImageFromJpg(idx)
    token_for_his = processor(raw_image, "his", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    token_for_her = processor(raw_image, "her", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    token_for_their = processor(raw_image, "their", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    inputs = processor(raw_image, text_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        if "logits" in outputs.keys():
            logits = outputs["logits"][0,-1,:]
        else:
            logits = outputs["decoder_logits"][0,-1,:]

    logits_for_his = logits[token_for_his] 
    logits_for_her = logits[token_for_her] 
    logits_for_their = logits[token_for_their]
    print(token_for_his,token_for_her,token_for_their)
    print({"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()})
    return {"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()}


def instruct_blipv2_get_probabilities_his_her_their(image_url: str, idx: str, system_content:str, usr_content:str,\
                                            text_input: str, model, processor) -> List:

    # raw_image = get_image(image_url)
    raw_image = readImageFromJpg(idx)
    token_for_his = processor(raw_image, "his", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    token_for_her = processor(raw_image, "her", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    token_for_their = processor(raw_image, "their", return_tensors="pt").to("cuda")["input_ids"][0][-1]
    if system_content:
        text = f'''{system_content} {usr_content} Description:{text_input}'''
    else:
        text = f'''{usr_content} Description:{text_input}'''
    inputs = processor(raw_image, text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        if "logits" in outputs.keys():
            logits = outputs["logits"][0, -1, :]
        else:
            logits = outputs["decoder_logits"][0, -1, :]

    logits_for_his = logits[token_for_his]
    logits_for_her = logits[token_for_her]
    logits_for_their = logits[token_for_their]
    print(token_for_his, token_for_her, token_for_their)
    print({"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()})
    return {"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()}


def blip_conditional_image_captioning(image_url: str, text_input: str, model, processor):
    """
    Returns the caption as output by the BLIP model. The output is based on an input-starting sentence.
    The function prints the caption, and returns the encoded tokens

    Args:
        image_url: url to the image to be captioned
        text_input: start sentence which includes the profession

    Returns:
        tensor (not decoded) with tokens of the caption
    
    """
    raw_image = get_image(image_url)

    inputs = processor(raw_image, text_input, return_tensors="pt").to(device)

    output = model.generate(**inputs)
    # print("CONDITIONAL", processor.decode(output[0], skip_special_tokens=True))
    print("CONDITIONAL", processor.decode(output[0], skip_special_tokens=True))

    return output

