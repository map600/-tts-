import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    print(f"正在下载 {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 下载文件
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        print(f"{filename} 下载完成!")
        return True
    except Exception as e:
        print(f"下载 {filename} 时出错: {e}")
        return False

# 创建输出目录
output_dir = "./output/xtts_train/XTTS_v2.0_original_model_files/"
os.makedirs(output_dir, exist_ok=True)

# 模型文件链接
files_to_download = {
    "dvae.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
    "mel_stats.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
    "vocab.json": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
    "model.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
}

# 下载文件
for filename, url in files_to_download.items():
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        success = download_file(url, filepath)
        if not success:
            print(f"无法下载 {filename}，请检查网络连接。")
    else:
        print(f"{filename} 已存在，跳过下载。")
        
print("所有文件检查完毕!")