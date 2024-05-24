from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

# Set up
model_cache_dir = "/usr/prakt/s0070/vlm-based-image-search/models"
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=model_cache_dir)

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=model_cache_dir, load_in_4bit=True, device_map="auto")

class Caption(BaseModel):
    caption: str = Field(description="Caption of the image")
parser = PydanticOutputParser(pydantic_object=Caption)

# Run
# prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)

image_path = "/usr/prakt/s0070/vlm-based-image-search/data/coco/images/val2017/000000397133.jpg"
image = Image.open(image_path)
prompt = """[INST] <image>\nWrite a caption for the photo. Consider the output template when you respond. Do not generate anything else. Here is the output template:
    {"caption": "short description of the photo"} Take a deep breath and answer only with a JSON. [/INST]"""

inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=200)

decoded_output = processor.decode(output[0], skip_special_tokens=True)
print(decoded_output)

parsed_output = parser.parse(decoded_output.split("[/INST]")[1].strip()).caption
# Save to a txt file
with open("/usr/prakt/s0070/vlm-based-image-search/outputs/output.txt", "w") as file:
    file.write(parsed_output)
