from huggingface_hub import snapshot_download

local_dir = './Qwen3-0.6B-Base'
snapshot_download(repo_id="Qwen/Qwen3-0.6B-Base", repo_type="model", local_dir=local_dir, force_download=True)