import os
import shutil

# 文件路径
file_path = r"E:\Anaconda\envs\myenv\Lib\site-packages\trainer\generic_utils.py"
backup_path = r"E:\Anaconda\envs\myenv\Lib\site-packages\trainer\generic_utils.py.bak"

# 创建备份
shutil.copy(file_path, backup_path)

# 读取原文件内容
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找并修改get_git_branch函数
in_git_branch_function = False
modified_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # 检查是否进入get_git_branch函数
    if line.strip().startswith('def get_git_branch():'):
        in_git_branch_function = True
        modified_lines.append(line)
        # 跳过原来的函数体
        i += 1
        # 跳过缩进的行直到下一个非缩进行
        while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
            i += 1
        
        # 添加新的函数实现
        modified_lines.append('    import subprocess\n')
        modified_lines.append('    try:\n')
        modified_lines.append('        out = subprocess.check_output(["git", "branch"]).decode("utf8")\n')
        modified_lines.append('        # 修复StopIteration异常问题\n')
        modified_lines.append('        current = next((line for line in out.split("\\n") if line.startswith("*")), "")\n')
        modified_lines.append('        if current:\n')
        modified_lines.append('            current = current.replace("* ", "")\n')
        modified_lines.append('        else:\n')
        modified_lines.append('            current = "unknown"\n')
        modified_lines.append('    except subprocess.CalledProcessError:\n')
        modified_lines.append('        current = "inside_docker"\n')
        modified_lines.append('    except FileNotFoundError:\n')
        modified_lines.append('        current = "unknown"\n')
        modified_lines.append('    return current\n')
        in_git_branch_function = False
        continue
    
    # 如果不在get_git_branch函数中，则保留原行
    if not in_git_branch_function:
        modified_lines.append(line)
    
    i += 1

# 写入修改后的内容
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(modified_lines)

print("Successfully modified generic_utils.py")