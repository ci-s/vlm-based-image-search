from typing import List, Dict

from services.faiss_service import FaissService

class ImageCaptions:
    """
    A class representing a collection of image captions.

    Attributes:
        filename_caption_dict (Dict[str, str]): A dictionary mapping filenames to captions.
        filenames (List[str]): A list of filenames.
        captions (List[str]): A list of captions.

    Methods:
        __init__(self, filename_caption_dict: Dict[str, str | List[str]], url_suffix: str | None) -> None: Initializes the ImageCaptions object.
        get_filenames(self, indices: List[int] = None) -> List[str]: Returns a list of filenames.
        get_captions(self, indices: List[int] = None) -> List[str]: Returns a list of captions.
        get_urls(self, indices: List[int] = None) -> List[str]: Returns a list of URLs corresponding to the filenames.
    """
    def __init__(self, filename_caption_dict: Dict[str, str], url_suffix: str | None = None) -> None:
        """
        filename_caption_dict is a dictionary with the following structure:
        {
            "filename1": "caption1",
            "filename2": "caption2",
            ...
        }
        """
        self.filename_caption_dict = filename_caption_dict
        self.filenames = list(self.filename_caption_dict.keys())
        self.captions = list(self.filename_caption_dict.values())
        
        self.url_suffix = url_suffix
        
    def get_filenames(self, indices: List[int] = None) -> List[str]:
        if indices is None:
            return list(self.filenames)
        return [self.filenames[i] for i in indices]
    
    def get_captions(self, indices: List[int] = None) -> List[str] | List[List[str]]:
        if indices is None:
            return self.captions
        return [self.captions[i] for i in indices]

    def get_filename_by_caption(self, caption: str) -> str:
        for filename, captions in self.filename_caption_dict.items():
            if caption in captions:
                return filename
        return None

    def get_caption_by_filename(self, filename: str) -> str:
        return self.filename_caption_dict.get(filename, None)

    def get_urls(self, indices: List[int] = None) -> List[str]:
        if self.url_suffix is None:
            raise ValueError("URL suffix is not set.")
        if indices is None:
            return [self.url_suffix + filename for filename in self.filenames]
        return [self.url_suffix + self.filenames[i] for i in indices]
    

class SearchService:
    def __init__(
        self, encoder_model: str, image_captions: ImageCaptions, k: int = 10
    ) -> None:
        self.faiss_service = FaissService(encoder_model)
        self.image_captions = image_captions
        self.faiss_service.create_index(self.image_captions.get_captions())
        self.k = k

    def search(self, query: str, return_url: bool = False) -> List[str]:
        D, I = self.faiss_service.search_index(query, self.k)
        
        if return_url:
            return self.image_captions.get_urls(I[0])[::-1] # Reverse the order
        else:
            return self.image_captions.get_filenames(I[0])[::-1] # Reverse the order
