from openai import OpenAI
from config.config import modelMap
import os
import time

os.environ["OPENAI_API_KEY"] = modelMap['GPT']['key']
os.environ["OPENAI_BASE_URL"] = modelMap['GPT']['api']


def gpt(model, user_content, system_content="", system=False):

    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system",
                         "content": system_content}),
    messages.append({"role": "user", "content": user_content})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    print(response)
    return response

def call_gpt(model, user_content, system=False, system_content=""):
    flag = 1
    while(flag):
        try:
            time.sleep(0.1)
            response = gpt(model=model, user_content=user_content, system=system, system_content=system_content)
            data = response.choices[0]
            res = data.message.content
            flag = 0
            return res
        except Exception as e:
            print("gpt call error: ", e)
            time.sleep(1)
    return 'gpt call error'

if __name__ == "__main__":
    res = call_gpt(model='gpt-4-turbo-preview', user_content='hello')
    print(res)