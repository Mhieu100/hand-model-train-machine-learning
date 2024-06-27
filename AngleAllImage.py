from PIL import Image
import os
import pandas as pd

# Đường dẫn đến thư mục chứa các ảnh gốc
source_folder = r"data"

# Đường dẫn đến thư mục lưu các ảnh đã xoay
destination_folder = r"data-angle"

# Đảm bảo thư mục đích tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Đường dẫn đến tệp Excel chứa thông tin góc xoay
excel_path = r"hands.xlsx"

# Đọc tệp Excel
df = pd.read_excel(excel_path)

# Chuyển đổi cột 'filename' thành chuỗi
df['ID'] = df['ID'].astype(str)

# Lặp qua các hàng trong DataFrame
for index, row in df.iterrows():
    filename = row['ID']+ '.png'  # Giả sử cột tên tệp là 'filename'
    angle = row['angle']  # Giả sử cột góc xoay là 'angle'
    # Tạo đường dẫn đầy đủ đến tệp ảnh
    file_path = os.path.join(source_folder, filename)

    # Kiểm tra xem tệp có phải là một ảnh không (dựa vào phần mở rộng)
    if filename.lower().endswith(('.png', '.jpg')):
        # Mở ảnh
        img = Image.open(file_path)

        # Xoay ảnh
        rotate_img = img.rotate(angle, expand=True)

        # Tạo đường dẫn đầy đủ đến tệp đích
        output_path = os.path.join(destination_folder, filename)

        # Lưu ảnh đã xoay vào thư mục đích
        rotate_img.save(output_path)

        print(f"Đã xoay và lưu: {filename} với góc {angle} độ")

print("Hoàn thành xoay tất cả ảnh!")
