import sys
sys.path.append("../")

from services.search import ImageRepresentations, SearchService

# This file is for the demo time. Since it is not possible to create captions from scratch during demo, at least for Llava (and maybe GIT model?), I assume there are already predicted captions. 
# For CLIP, it is possible to create embeddings from scratch or load from saved as you wish.

###################################
# Search params                   #
###################################
query = "birthday cake"
k = 10
threshold = 0.3

###################################
# Llava & GIT example             #
###################################
from sentence_transformers import SentenceTransformer
import json

caption_path = "/usr/prakt/s0070/vlm-based-image-search/outputs/response_dict.json" # format-> filename:caption #TODO: Change this path Emine
predicted_file = json.load(open(caption_path))

image_representions = ImageRepresentations(filenames=list(predicted_file.keys()), representations=list(predicted_file.values()), url_prefix="http://images.cocodataset.org/val2017/")
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

###################################
# CLIP example                   #
###################################
# might require rework
from torch.utils.data import DataLoader

from services.vlm_models_service import SearchModel
from data.get_coco import load_cocos_like_dataset_in_range

# either load from saved (i don't know how) or create from scratch
model_name = "CLIP"
model = SearchModel(model_name) # or load from saved
batch_size = 16
coco_dataset = load_cocos_like_dataset_in_range(5, model_name, last_index=16, option="concat")
dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=False)
image_embeddings, text_embeddings = model.encode_images_texts(dataloader) # or encode_images directly
filenames = None # TODO: Get filenames from dataloader/cocodataset I think you already have a to do this Emrah, so not spending time on it
image_representions = ImageRepresentations(filenames=filenames, representations=image_embeddings, url_prefix="http://images.cocodataset.org/val2017/")
encoder_model = None
query = model.encode_text_list([query]) # this can also be embedded into FaissService via providing "model" as encoder model, lmk if that makes more sense or go ahead and modify it

###################################
# Search                         #
###################################
ss = SearchService(image_representations=image_representions, encoder_model=encoder_model, threshold=threshold)

# After creating the demo, we'll probably have an endpoint for the part below, so that we don't need to load models again for each request.
# But for the sake of testing, I'll add one query and search for it
retrieved_files = ss.search(query, return_url=True)
print("Query: ", query)
for i, file in enumerate(retrieved_files):
    print(f"{i+1}. {file}")