from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root_dir: str
    data_dir: str
    output_dir: str
    log_dir: str
    
load_dotenv()
settings = Settings()
