import os
import re
import shutil

def copy_files_above_prob(source_dir, target_dir, threshold=50):
    """
    遍历 source_dir，将文件名中 prob_<数字> 大于 threshold 的文件复制到 target_dir，
    并保持目录结构。

    Args:
        source_dir (str): 源文件夹路径
        target_dir (str): 目标文件夹路径（会自动创建）
        threshold (int): 概率阈值，默认 50
    """
    # 确保目标根目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 正则提取 prob_ 后面的数字（要求数字前后非字母，但为了安全只匹配数字）
    # 文件名示例: channel_4_prob_94.csv
    pattern = re.compile(r'prob_(\d+)')  # 捕获 prob_ 后的数字

    matched_count = 0
    copied_count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 只处理 .csv 文件（可根据需要修改扩展名）
            if not file.lower().endswith('.csv'):
                continue

            # 提取 prob_ 后面的数字
            match = pattern.search(file)
            if not match:
                continue   # 没有 prob_xx 字段，跳过

            prob_value = int(match.group(1))
            if prob_value <= threshold:
                continue   # 不大于阈值，跳过

            # 构建相对路径和目标完整路径
            rel_path = os.path.relpath(root, source_dir)
            target_subdir = os.path.join(target_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)

            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_subdir, file)

            # 复制文件（保留原时间戳等元数据）
            shutil.copy2(src_file, dst_file)
            copied_count += 1
            print(f"复制: {src_file} -> {dst_file} (prob={prob_value})")

        matched_count += len([f for f in files if pattern.search(f)])

    print(f"\n✅ 完成！共找到 {matched_count} 个符合命名模式的文件，其中 prob > {threshold} 的文件复制了 {copied_count} 个。")

if __name__ == "__main__":
    # ===== 这里修改为你的实际路径 =====
    source_dir = r"F:\\my_output_new_DC"  # 源文件夹
    target_dir = r"F:\\my_output_new_DC_select"   # 目标文件夹

    copy_files_above_prob(source_dir, target_dir, threshold=50)