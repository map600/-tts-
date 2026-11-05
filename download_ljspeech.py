import os
import tarfile
import random
import requests
import time

def download_with_resume(url, filename):
    """使用requests库下载文件，支持断点续传和进度显示"""
    print(f"\n开始下载: {url}")
    print(f"保存到: {filename}")
    
    # 检查是否已存在部分文件
    resume_byte_pos = 0
    if os.path.exists(filename):
        resume_byte_pos = os.path.getsize(filename)
        print(f"检测到已存在部分文件 ({resume_byte_pos / (1024*1024):.2f} MB)")
        print("将尝试断点续传...")
    
    headers = {'Range': f'bytes={resume_byte_pos}-'}
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    try:
        response = session.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # 获取文件总大小
        content_length = response.headers.get('content-length')
        if content_length is None:
            print("警告: 无法获取文件总大小")
            total_size = None
        else:
            total_size = int(content_length) + resume_byte_pos
            print(f"文件总大小: {total_size / (1024*1024):.2f} MB")
        
        # 开始下载
        start_time = time.time()
        with open(filename, 'ab' if resume_byte_pos else 'wb') as f:
            downloaded = resume_byte_pos
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 显示进度
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        elapsed_time = time.time() - start_time
                        speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                        
                        print(f"\r下载进度: {percent:.2f}% | {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB | 速度: {speed / (1024*1024):.2f} MB/s | 用时: {elapsed_time:.1f}s", end="")
                    else:
                        elapsed_time = time.time() - start_time
                        speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                        print(f"\r已下载: {downloaded / (1024*1024):.2f} MB | 速度: {speed / (1024*1024):.2f} MB/s | 用时: {elapsed_time:.1f}s", end="")
        
        print("\n下载完成!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n下载出错: {e}")
        return False

# 主程序
def main():
    download_url = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    output_dir = "./datasets"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载文件
    tar_path = os.path.join(output_dir, "LJSpeech-1.1.tar.bz2")
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        if download_with_resume(download_url, tar_path):
            break
        retry_count += 1
        if retry_count < max_retries:
            print(f"\n正在重试下载... ({retry_count}/{max_retries})")
            time.sleep(5)
        else:
            print(f"\n下载失败，已达到最大重试次数 ({max_retries})")
            return False
    
    # 验证文件大小（可选）
    expected_size = 2748572632  # 预期大小（字节）
    actual_size = os.path.getsize(tar_path)
    print(f"\n文件大小验证: {actual_size / (1024*1024):.2f} MB / {expected_size / (1024*1024):.2f} MB")
    
    if actual_size < expected_size * 0.95:  # 允许5%的误差
        print("警告: 文件大小可能不完整!")
        # 不中断程序，继续尝试解压
    
    # 解压文件
    print("\n正在解压...")
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(output_dir)
        print("解压完成!")
    except tarfile.ReadError as e:
        print(f"解压出错: {e}")
        print("可能是下载文件损坏，建议重新下载")
        return False
    
    # 创建训练/验证集拆分
    data_dir = os.path.join(output_dir, "LJSpeech-1.1")
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata_train_path = os.path.join(data_dir, "train.csv")
    metadata_val_path = os.path.join(data_dir, "val.csv")
    
    print("\n正在创建训练/验证集拆分...")
    
    # 读取metadata.csv
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 打乱顺序
        random.shuffle(lines)
        
        # 创建训练集和验证集
        train_lines = lines[:12000]
        val_lines = lines[12000:13100]
        
        # 保存训练集和验证集
        with open(metadata_train_path, "w", encoding="utf-8") as f:
            f.writelines(train_lines)
        
        with open(metadata_val_path, "w", encoding="utf-8") as f:
            f.writelines(val_lines)
        
        print(f"训练集: {len(train_lines)} 条记录")
        print(f"验证集: {len(val_lines)} 条记录")
        
    except Exception as e:
        print(f"创建训练/验证集时出错: {e}")
        return False
    
    # 删除压缩文件
    print("\n正在清理压缩文件...")
    try:
        os.remove(tar_path)
        print(f"已删除: {tar_path}")
    except Exception as e:
        print(f"删除压缩文件时出错: {e}")
        # 不中断程序
    
    print("\n数据集准备完成!")
    print(f"数据集位置: {data_dir}")
    return True

if __name__ == "__main__":
    if not main():
        print("\n程序执行失败!")
        exit(1)