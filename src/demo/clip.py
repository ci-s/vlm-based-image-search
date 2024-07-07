import open_clip
import torch

model_id = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
model, _ = open_clip.create_model_from_pretrained(model_id)
tokenizer = open_clip.get_tokenizer(model_id)
model.eval()
text_encoder = lambda query : model.encode_text(tokenizer(query))

@torch.no_grad()
def get_text_embedding(query):
    text_feat = text_encoder(query)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return text_feat