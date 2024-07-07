import os
import json
import urllib.request
import zipfile
import sys
sys.path.append("../")
from services.settings import settings

class VisualGenomeDataset:
    def __init__(self, data_dir,size):
        self.data_dir = data_dir
        self.size = size
        #paths
        self.object_zip_file_path = os.path.join(data_dir, 'objects.json.zip')
        self.object_json_file_path = os.path.join(data_dir, 'objects.json')
        self.metadata_zip_file_path = os.path.join(data_dir, 'image_data.json.zip')
        self.metadata_json_file_path = os.path.join(data_dir, 'image_data.json')
        # create / download / load necessary data
        self.init_data()
        
    def init_data(self):
        # Ensure the data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Download and unzip the object file if not present
        if not os.path.exists(self.object_json_file_path):
            self._download_objects()
            
        # Load the object json data
        with open(self.object_json_file_path, 'r') as file:
            print("Loading object data...")
            self.object_data = json.load(file)
                
        # Download and unzip metadata if not present
        if not os.path.exists(self.metadata_json_file_path):
            self._download_metadata()
                
        # Load the metadata json data
        with open(self.metadata_json_file_path, 'r') as file:
            print("Loading metadata...")
            self.metadata = json.load(file)

    def _download_objects(self):
        url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"
        print("Downloading object json...")
        urllib.request.urlretrieve(url, self.object_zip_file_path)
        
        print("Unzipping object json...")
        with zipfile.ZipFile(self.object_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("Removing the zip file...")
        os.remove(self.object_zip_file_path)
        
    def _download_metadata(self):
        url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
        print("Downloading metadata...")
        urllib.request.urlretrieve(url,self.metadata_zip_file_path)
        
        print("Unzipping metadata...")
        with zipfile.ZipFile(self.metadata_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print("Removing the zip file...")
        os.remove(self.metadata_zip_file_path)
        
    def get_image_url_from_id(self, image_id):
        return self.metadata[image_id - 1]['url'] 
            
    def get_filenames_and_objects(self):
        result = {}
        for entry in self.object_data[:self.size]:
            image_url = self.get_image_url_from_id(entry['image_id'])
            objects = [obj['names'][0] for obj in entry['objects']]
            result[image_url] = list(set(objects))
        return result
    
    def save_filenames_and_objects(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.get_filenames_and_objects(), file, indent=4)

# Example usage
data_dir = os.path.join(settings.data_dir, 'visual_genome')

vg_dataset = VisualGenomeDataset(data_dir, size=1000) # size = no. of images to load
filenames_and_objects = vg_dataset.get_filenames_and_objects()
vg_dataset.save_filenames_and_objects(os.path.join(data_dir, 'filenames_and_objects.json'))

# Print a few examples
for image_id, objects in list(filenames_and_objects.items())[:3]:
    print(f"Image URL: {image_id}")
    print(f"Objects: {objects}")
    print("-----")
