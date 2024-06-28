import sys
sys.path.append("../")
from services.get_coco import load_cocos_like_dataset
from services.vlm_service import EmbeddingService
from torch.utils.data import DataLoader

model_name = "CLIP"
batch_size = 64
coco_dataset = load_cocos_like_dataset(5, model_name)
embedding_service = EmbeddingService(model_name, coco_dataset, "mscoco")
embedding_service.get_embeddings(batch_size)

