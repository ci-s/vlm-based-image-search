import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import requests
import os
import sys
sys.path.append("../")
from services.settings import settings

# Load GIT model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
K = 5000

def generate_caption(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

def fetch_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    return image

def get_image_urls(img_ids, coco):
    img_urls = []
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        img_urls.append(img['coco_url'])
    return img_urls

def embed_text(text,embedder):
    return embedder.encode([text])

def set_coco_object(data_dir):
    ann_file = os.path.join(data_dir, 'annotations/captions_val2017.json')
    coco = COCO(ann_file)
    return coco


coco = set_coco_object(os.path.join(settings.data_dir,'coco-images'))
GT_img_ids = [int(line.strip()) for line in open(os.path.join(settings.data_dir,'coco_embeddings/coco_image_ids_' + str(K) +'.txt'))]

image_urls = get_image_urls(GT_img_ids, coco)
  
generated_captions = []
generated_caption_embeddings = []
for counter,img_url in enumerate(image_urls):
    if counter % 100 == 0:
        print(f'Processing image {counter}')
    image = fetch_image(img_url)
    caption = generate_caption(image)
    generated_captions.append(caption)
    
with open(os.path.join(settings.data_dir,'git_embeddings/git_captions_' + str(K) +'.txt'), 'w') as f:
    for caption in generated_captions:
        f.write(caption + '\n')