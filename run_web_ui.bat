@echo off
chcp 65001 >nul
cd /d "e:\大三下机器学习\pythonProject1"
call conda activate myenv
start "" http://127.0.0.1:8000
python web_tts_server.py
pause
