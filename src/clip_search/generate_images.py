import os
import json
from diffusers import StableDiffusionPipeline
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm
import torch
import sys 
sys.path.append("..")
from data.get_coco import load_cocos_like_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
print(device)
pipeline = pipeline.to(device)  
clip_model = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
model, processor = create_model_from_pretrained(clip_model)
tokenizer = get_tokenizer(clip_model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
checkpoint_root = "../../data/nocaps/"
images_output_dir = os.path.join(checkpoint_root, 'generated_images')
os.makedirs(images_output_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_root, "checkpoint.json")

def get_start_index(checkpoint_path = checkpoint_path):
    start_index = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            start_index = checkpoint.get('last_processed_index', 0)
            print(f"Resuming from checkpoint, starting at index {start_index}")
    return start_index

def get_captions_from_dataset(dataset):
    captions_list = []
    for _, captions in dataset:
        captions_list.extend([caption for caption in captions])
    return captions_list

def generate_image(caption):
   image = pipeline(caption).images[0]
   return image

def save_image(image, index, output_dir = images_output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(images_output_dir, exist_ok=True)
    image_filename = f'image_{index}.png'
    image_path = os.path.join(output_dir, image_filename)
    image.save(image_path)

@torch.no_grad()
def generate_image_batch(dataset, checkpoint_size = 100):
    captions = get_captions_from_dataset(dataset)
    start_index = get_start_index()
    len_captions = len(captions)
    img_embeddings = []
    for ind in tqdm(range(start_index, len_captions), desc="Generating image"):
        image = generate_image(captions[ind])
        save_image(image, ind)
        img_embeddings.append(model.encode_image(processor(image).unsqueeze(0).to(device)))

@torch.no_grad()
def generate_first_image_batch(dataset, checkpoint_size = 100):
    captions = get_captions_from_dataset(dataset)
    captions_per_image = len(captions) // len(dataset)
    start_index = get_start_index
    len_captions = len(captions)    
    for ind in tqdm(range(start_index, len_captions), desc="Generating image"):
        index = ind * captions_per_image
        image = generate_image(captions[index])
        save_image(image, index)

# dataset = load_cocos_like_dataset(5, "CLIP")
# generate_image_batch(dataset)

img_root = os.path.join(checkpoint_root, "validat_images")
caption_path = os.path.join(checkpoint_root, "no_caps.json")
dataset = load_cocos_like_dataset(10, "CLIP", img_root, caption_path)
generate_first_image_batch(dataset)
        
        
        
            
    