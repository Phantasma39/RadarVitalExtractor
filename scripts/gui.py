# 启动雷达数据处理可视化界面
#
# 用法: python scripts/gui.py
import _bootstrap  # noqa: F401  加入 src/ 到 sys.path + 修正控制台编码

from radar_project.gui import main

if __name__ == "__main__":
    main()
