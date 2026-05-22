from huggingface_hub import snapshot_download

local_dir = './clip'
snapshot_download(repo_id="openai/clip-vit-base-patch32", repo_type="model", local_dir=local_dir, force_download=True)