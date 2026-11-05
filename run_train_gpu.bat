@echo off
chcp 65001 >nul
cd /d "e:\大三下机器学习\pythonProject1"
call conda activate myenv
python train_xtts_resume.py
pause
