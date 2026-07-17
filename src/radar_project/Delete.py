import os
import re

# 根目录（存放所有人的文件夹）
root_dir = r"F:\my_output_new"

# 正则：提取 prob 后面的数值
pattern = re.compile(r"prob_(\d+)", re.IGNORECASE)

deleted_files = 0

for person_folder in os.listdir(root_dir):
    person_path = os.path.join(root_dir, person_folder)

    # 只处理文件夹
    if not os.path.isdir(person_path):
        continue

    for file in os.listdir(person_path):
        if not file.endswith(".csv"):
            continue

        match = pattern.search(file)
        if match:
            prob_value = int(match.group(1))

            if prob_value < 50:
                file_path = os.path.join(person_path, file)
                os.remove(file_path)
                print(f"删除文件: {file_path}")
                deleted_files += 1

print(f"\n完成，共删除 {deleted_files} 个文件")