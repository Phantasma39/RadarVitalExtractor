import os
import sys

# ====================== 路径配置 ======================
# 也可以从命令行覆盖：python -m radar_project.rename <文件夹路径> [旧片段] [新片段]
folder_path = r"F:\gaoxiangrong"   # 改成你的文件夹路径
old_pattern = "_gaoxiangrong_Raw_"
new_pattern = "_Raw_gaoxiangrong_"
# ======================================================

if len(sys.argv) >= 2:
    folder_path = sys.argv[1]
if len(sys.argv) >= 4:
    old_pattern = sys.argv[2]
    new_pattern = sys.argv[3]


def rename_in_folder(folder_path, old_pattern, new_pattern):
    """
    把文件名中的 old_pattern 批量替换为 new_pattern。

    返回实际重命名的文件数。
    """
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        print("用法: python -m radar_project.rename <文件夹路径> [旧片段] [新片段]")
        return 0

    renamed = 0
    # 先收集再改名，避免 os.listdir 在做重命名时遍历结果不稳定
    for filename in sorted(os.listdir(folder_path)):
        old_path = os.path.join(folder_path, filename)
        print(filename)

        # 只处理文件，跳过文件夹
        if not os.path.isfile(old_path):
            continue

        new_filename = filename.replace(old_pattern, new_pattern)
        new_path = os.path.join(folder_path, new_filename)

        # 只有名字不一样才重命名，避免报错
        if new_filename != filename:
            if os.path.exists(new_path):
                print(f"跳过（目标已存在）：{new_filename}")
                continue
            os.rename(old_path, new_path)
            renamed += 1
            print(f"已改名：{filename} → {new_filename}")

    print(f"批量改名完成！共重命名 {renamed} 个文件")
    return renamed


if __name__ == "__main__":
    rename_in_folder(folder_path, old_pattern, new_pattern)
