from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_tree(
    "allenai/dolma3_dolmino_mix-100B-1125", 
    repo_type="dataset", 
    recursive=True
)

total_size = 0
file_count = 0
for f in files:
    if hasattr(f, 'size') and f.size is not None:
        total_size += f.size
        file_count += 1

print(f"文件数量: {file_count}")
print(f"总大小: {total_size / (1024**3):.2f} GB")
print(f"总大小: {total_size / (1024**4):.2f} TB") 
