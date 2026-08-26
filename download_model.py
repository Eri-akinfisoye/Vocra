from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print("Downloading Chatterbox model files...")
print("Downloads resume automatically if interrupted")

snapshot_download(
    repo_id="resemble-ai/chatterbox",
    local_dir="./models/chatterbox",
    token=HF_TOKEN
)

print("Download complete!")