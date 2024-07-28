from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
import os
import json
import sys

sys.path.append("../")
from services.settings import settings

# Set up
model_cache_dir = "/usr/prakt/s0070/vlm-based-image-search/models"
output_file = "/usr/prakt/s0070/vlm-based-image-search/outputs/response_dict_keyword_500.json"

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=model_cache_dir)
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=model_cache_dir, load_in_4bit=True, device_map="auto")

class Caption(BaseModel):
    caption: list[str] = Field(description="Caption of the image")
parser = PydanticOutputParser(pydantic_object=Caption)

prompt = """[INST] <image>\nWrite a limited series of unique keywords for the photo as caption. Keywords should contain objects and actions. Consider the output template when you respond. Do not generate anything else. Here is the output template:
    {"caption": "series of unique keywords"} Take a deep breath and answer only with a JSON. [/INST]"""
    
response_dict = {}

counter = 0
# Run
file_restriction = os.path.join(settings.data_dir, "coco/annotations/captions_val2017.json")
file_restriction = json.load(open(file_restriction))

file_list = []
for d in file_restriction["images"][:500]:
    filename = d["file_name"]
    image_path = os.path.join("/usr/prakt/s0070/vlm-based-image-search/data/coco/images/val2017", filename)
    image = Image.open(image_path)


    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=200)

    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    print(decoded_output)

    parsed_output = parser.parse(decoded_output.split("[/INST]")[1].strip()).caption
    
    response_dict[filename] = parsed_output


with open(output_file, 'w') as fp:
    json.dump(response_dict, fp)