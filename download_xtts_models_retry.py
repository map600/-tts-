import requests
import os
from tqdm import tqdm
import time

def download_file_with_retry(url, filename, max_retries=3, timeout=30):
    """
    下载文件并显示进度条，带重试机制
    """
    print(f"正在下载 {filename}...")
    
    # 创建目录
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 如果文件已存在，直接返回
    if os.path.exists(filename):
        print(f"{filename} 已存在，跳过下载。")
        return True
    
    for attempt in range(max_retries):
        try:
            print(f"尝试第 {attempt + 1} 次下载...")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
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
            if attempt < max_retries - 1:
                print(f"等待10秒后重试...")
                time.sleep(10)
            else:
                print(f"已达到最大重试次数，下载失败。")
    
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
failed_downloads = []
for filename, url in files_to_download.items():
    filepath = os.path.join(output_dir, filename)
    success = download_file_with_retry(url, filepath, max_retries=3, timeout=60)
    if not success:
        failed_downloads.append(filename)
        
if failed_downloads:
    print(f"\n以下文件下载失败:")
    for filename in failed_downloads:
        print(f"- {filename}")
    print("\n请检查网络连接，或稍后重试。")
else:
    print("\n所有文件下载完成!")