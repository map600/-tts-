#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接运行GUI程序
try:
    from real_xtts_gui import main
    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"运行XTTS GUI时出错: {e}")
    input("按回车键退出...")