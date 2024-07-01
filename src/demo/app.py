# run python app.py or gradio app.py (to reload the app on changes automatically)

import gradio as gr
from sentence_transformers import SentenceTransformer
import json
import os
import sys
sys.path.append("../")

from services.search import ImageRepresentations, SearchService
from services.settings import settings

coco_path = os.path.join(settings.data_dir, "coco/images/val2017/")

llava_caption_path = os.path.join(settings.output_dir, "response_dict.json") # format-> filename:caption #TODO: Create one for git too
llava_predicted_file = json.load(open(llava_caption_path))

llava_image_representions = ImageRepresentations(filenames=list(llava_predicted_file.keys()), representations=list(llava_predicted_file.values()), url_prefix="http://images.cocodataset.org/val2017/")
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

llava_ss = SearchService(image_representations=llava_image_representions, encoder_model=encoder_model, k=10) #threshold=0.3

# define another search service i.e for CLIP, change image representations
clip_ss = SearchService(image_representations=llava_image_representions, encoder_model=encoder_model, k=10) # TODO
git_ss = SearchService(image_representations=llava_image_representions, encoder_model=encoder_model, k=10) # TODO

# We can customize this function to use different models
def llava_search(query):
    retrieved_files, captions = llava_ss.search_and_caption(query, return_url=False)
    return [(os.path.join(coco_path, file), caption) for file, caption in zip(retrieved_files, captions)]

def git_search(query):
    retrieved_files, captions = git_ss.search_and_caption(query, return_url=False)
    return [(os.path.join(coco_path, file), caption) for file, caption in zip(retrieved_files, captions)]

def clip_search(query):
    retrieved_files = clip_ss.search(query, return_url=False)
    return [(os.path.join(coco_path, file), None) for file in retrieved_files] # No captions for this option

def search(model_name, query):
    if model_name == "LLaVa":
        return llava_search(query)
    elif model_name == "CLIP":
        return clip_search(query)
    elif model_name == "GIT":
        return git_search(query)
    
demo = gr.Interface(
    fn=search,
    inputs=[gr.Radio(choices=["LLaVa", "CLIP", "GIT"]), gr.Textbox(label="Enter your query here")], #, gr.Slider(0, 1, step=0.1, label="Similarity threshold"), gr.Number(value=5, label="Top k") # these can be set as parameters for the search function, requires change in SearchService
    outputs=[gr.Gallery(label="Retrieved images")],
)

demo.launch()
