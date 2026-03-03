from src.template_generator_utils import load_metadata_to_dict
from tqdm import tqdm
import os, requests
from PIL import Image
from io import BytesIO


def main():
    metadata_dict = load_metadata_to_dict('data/visogender_data/OO/OO_Visogender_02102023.tsv', 'OO')
    for IDX_dict in metadata_dict:
        for metadata_key in tqdm(IDX_dict):
            url = IDX_dict[metadata_key]["url"]
            if os.path.exists(f'imgs/{metadata_key}.jpg'):
                continue
            try:
                saveImg('imgs',url,metadata_key)
            except Exception as e:
                print('error:', metadata_key, url, e)

def saveImg(save_directory, url, idx):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    headers = {"User-Agent": "OxAI"}
    # r = requests.get(image_url, stream=True, headers=headers)
    response = requests.get(url, stream=True, headers=headers)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        save_path = os.path.join(save_directory, f'{idx}.jpg')
        image.save(save_path)
        print(f'图片已保存到 {save_path}')
    else:
        print(idx, url, '图片下载失败，状态码：', response.status_code)

if __name__ == '__main__':
    main()