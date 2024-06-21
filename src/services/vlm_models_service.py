from abc import abstractmethod, ABC
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM
from torchvision import transforms
import sys
from tqdm import tqdm
sys.path.append("../")



class Caption(BaseModel):
    caption: str = Field(description="Caption of the image")
parser = PydanticOutputParser(pydantic_object=Caption)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Ensure this is the correct quant type for your use case
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_parameters_default = {
                    "CLIP" : {"models_id" : "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"},
                    "GIT" : {"models_id" : "microsoft/git-base",
                             "max_length" : 50},
                    "Llava" : {"models_id" : "llava-hf/llava-v1.6-mistral-7b-hf",
                               "prompt" : """[INST] <image>\nWrite a caption for the photo. Consider the output template when you respond. Do not generate anything else. Here is the output template:
                                            {"caption": "short description of the photo"} Take a deep breath and answer only with a JSON. [/INST]""",
                                "max_new_tokens" : 200,
                                "quantization_config" : quantization_config}}

class BaseVLM(ABC):
    def __init__(self, model_name, model_parameters = None):
        """
        Initialize the BaseVLM class with model name and parameters.
        
        Parameters:
        model_name (str): Name of the model.
        model_parameters (dict, optional): Parameters for the model. If None, default parameters are used.
        """
        
        if model_parameters is None:
            self.model_parameters = model_parameters_default[model_name]
        else: 
            self.model_parameters = model_parameters[model_name]
        self.model_name = model_name
        self.model_id = self.model_parameters["models_id"]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model, self.processor, self.image_model, self.text_model, self.image_processor, self.tokenizer = None, None, None, None, None, None
        self._initialize_models()
        
    def get_models(self):
        return self.model, self.processor, self.text_model, self.tokenizer
    
    @abstractmethod
    def _initialize_models(self):
        pass
    
    @abstractmethod
    def _process_images(self, images):
        pass
    
    @torch.no_grad
    def encode_images_texts(self, dataloader):
        """
        Encode both images and texts from the dataloader.
        
        Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and texts.
        
        Returns:
        (torch.Tensor, torch.Tensor): Encoded images and texts.
        """
        
        image_encodings = []
        text_encodings = []
        
        for images, texts in tqdm(dataloader, desc="Embedding"):
            images = images.to(self.device)
            texts = self._flatten_texts(texts)
            image_encodings.append(self._process_images(images))
            text_encodings.append(self.text_model(texts))
            
        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)

        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)
        
        return image_encodings, text_encodings
    
    @torch.no_grad
    def encode_images(self, dataloader):
        """
        Encode only images from the dataloader.
        
        Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images.
        
        Returns:
        torch.Tensor: Encoded images.
        """
        
        image_encodings = []
        
        for images, _ in tqdm(dataloader, desc="Embedding"):
            images = images.to(self.device)
            image_encodings.append(self._process_images())
                        
        image_encodings = torch.cat(image_encodings)
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        
        return image_encodings
    
    def _flatten_texts(self, texts):
        """
        Flatten a list of lists of texts into a single list.
        
        Parameters:
        texts (list of list of str): Nested list of texts.
        
        Returns:
        list of str: Flattened list of texts.
        """
        
        return [texts[j][i] for i in range(len(texts[0])) for j in range(len(texts))]
    
    @torch.no_grad
    def encode__texts(self, dataloader):
        """
        Encode only texts from the dataloader.
        
        Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of texts.
        
        Returns:
        torch.Tensor: Encoded texts.
        """
        
        text_encodings = []
        
        for _, texts in tqdm(dataloader, desc="Embedding"):
            texts = self._flatten_texts(texts)
            text_encodings.append(self.text_model(texts))
            
        text_encodings = torch.cat(text_encodings)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)
        
        return text_encodings
    
    def get_text_to_image_map(self, captions_per_image, length_images):
        """
        Create a mapping from texts to images based on the number of captions per image.
        
        Parameters:
        captions_per_image (int): Number of captions per image.
        length_images (int): Number of images.
        
        Returns:
        torch.LongTensor: Mapping from texts to images.
        """
        
        text_to_image_map = [i for i in range(length_images) for _ in range(captions_per_image)]
        text_to_image_map = torch.LongTensor(text_to_image_map).to(self.device)
        return text_to_image_map
    
class CLIPModel(BaseVLM):
    def _initialize_models(self):
        self.model, self.processor = create_model_from_pretrained(self.model_id)
        self.tokenizer = get_tokenizer(self.model_id)
        self.model.to(self.device).eval()
        self.image_processor = self.processor
        self.image_model = lambda images : self.model.encode_image(images)
        self.text_model = lambda texts : self.model.encode_text(self.tokenizer(texts).to(self.device))
        
    def _process_images(self, images):
        """
        Process a batch of images using the CLIP model.
        
        Parameters:
        images (torch.Tensor): Batch of images to be processed.
        
        Returns:
        torch.Tensor: Processed image encodings.
        """
        
        return self.image_model(images)
    
class CaptionModel(BaseVLM, ABC):
    def __init__(self, model_name, model_parameters=None):
        super().__init__(model_name, model_parameters)
        self.captions = []
    
    def encode_images_texts(self, dataloader):
        self.captions = []
        return super().encode_images_texts(dataloader) 
    
    def encode_images(self, dataloader):
        self.captions = []
        return super().encode_images(dataloader)   
    
    def _initiate_model_attributes(self, model, processor):
        """
        Initialize model attributes for captioning models.
        
        Parameters:
        model (torch.nn.Module): The model to be used for captioning.
        processor (transformers.Processor): The processor associated with the model.
        """
        self.model = model
        self.processor = processor
        self.processor.image_processor.do_resize = False
        text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model = lambda texts : text_model.encode(texts, convert_to_tensor=True).to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.image_model = lambda images : self._create_caption(images)
    
    @abstractmethod
    def _create_caption(self, images):
        """
        Abstract method to create captions for images.
        Must be implemented by subclasses.
        
        Parameters:
        images (torch.Tensor): Batch of images to be captioned.
        
        Returns:
        list of str: List of generated captions.
        """
        pass
    
    def get_captions(self):
        return self.captions
    
    def _process_images(self, images):
        captions = self.image_model(images)
        self.captions.extend(captions)
        return self.text_model(captions)

class LlavaModel(CaptionModel):
    def _initialize_models(self):
        processor = LlavaNextProcessor.from_pretrained(self.model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, torch_dtype =torch.float16, low_cpu_mem_usage=True, device_map="auto")
        self._initiate_model_attributes(model, processor)
    
    def _create_caption(self, images):
        caption_list = []
        for img in images:
            img = transforms.ToPILImage()(img)
            inputs = self.processor(self.model_parameters["prompt"], img, return_tensors = "pt").to(self.device)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model.generate(**inputs, max_new_tokens=self.model_parameters["max_new_tokens"])
            decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
            parsed_output = parser.parse(decoded_output.split("[/INST]")[1].strip()).caption
            caption_list.append(parsed_output)
        return caption_list

class GITModel(CaptionModel):
    def _initialize_models(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        processor.image_processor.do_rescale = False
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._initiate_model_attributes(model, processor)
        
    def _create_caption(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=self.model_parameters["max_length"])
        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens = True)
        caption_list = [generated_captions[i] for i in range(len(generated_captions))]
        return caption_list

    

        
    
    
    
        



         

        
        

