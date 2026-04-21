import os

# ====================== 你只需要改这里 ======================
folder_path = r"F:\gaoxiangrong"   # 改成你的文件夹路径
# ==========================================================

# 遍历文件夹里所有文件
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)
    print(filename)
    # 只处理文件，跳过文件夹
    if not os.path.isfile(old_path):
        continue

    # 替换规则：把 _AAA_Raw_ 替换成 _Raw_name_
    new_filename = filename.replace("_gaoxiangrong_Raw_", "_Raw_gaoxiangrong_")
    new_path = os.path.join(folder_path, new_filename)

    # 只有名字不一样才重命名，避免报错
    if new_filename != filename:
        os.rename(old_path, new_path)
        print(f"已改名：{filename} → {new_filename}")

print("✅ 批量改名完成！")