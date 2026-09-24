# 批量重命名脚本（薄包装，实际逻辑在 src/radar_project/rename.py）
#
# 用法：python scripts/rename.py <文件夹路径> [旧片段] [新片段]
#      不带参数时使用 src/radar_project/rename.py 顶部配置的默认路径
import _bootstrap  # noqa: F401  把 src/ 加入 sys.path，免安装即可 import

from radar_project.rename import rename_in_folder, folder_path, old_pattern, new_pattern

if __name__ == "__main__":
    rename_in_folder(folder_path, old_pattern, new_pattern)
