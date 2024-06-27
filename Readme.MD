Dự đoán hình ảnh tay trái tay phải bằng thuật toán Hồi Quy

Data ảnh : 
Tập train 1: https://drive.google.com/file/d/1hpUW7S3mBxVd4GdHWaFLG8hdPhoqk7Es/view?usp=sharing
Tập train 2: https://drive.google.com/file/d/1Kp19U3qbzI0wD-dlrHs2oepXLcTGsIp8/view?usp=sharing
Tập train 3: https://drive.google.com/file/d/1ymxaBEmegQC3uNZUhT9jWKdCYPpAWW1N/view?usp=sharing
Data gán nhãn :
https://drive.google.com/drive/u/1/folders/1OTjUnKEKdOnkkD7fd9YlZWly-b2uuLnF?fbclid=IwZXh0bgNhZW0CMTAAAR0bTzd_Ah8sXTvpLkrVyp5Zz6rRniqrp6G5D8PEJ0mOyCNgDwTL9xIhYh8_aem_cKouYY9_kjCLtOId2VFdqw

https://drive.google.com/drive/u/0/folders/171DqtcuT002A9dXhyGeBKA-uqBXqLvSF

1 Đánh nhãn bàn tay
1.1 MỤC TIÊU:
· Đánh nhãn hình ảnh bàn tay (trái, phải, 2 tay),
· Đánh nhãn góc xoay của bàn tay.
1.2 HƯỚNG DẪN ĐÁNH NHÃN TAY TRÁI/PHẢI/2 TAY:
· Đánh nhãn “trái” nếu ngón cái ở bên phải,
· Đánh nhãn “phải” nếu ngón cái ở bên trái,
· Đánh nhãn “hai tay” (cả 2 tay),
· Đánh nhãn “loại” trong trường hợp bàn tay bất thường (dị tật, thiếu ngón, cụt ngón, …).
1.3 HƯỚNG DẪN ĐÁNH NHÃN GÓC XOAY:
· Bàn tay tiêu chuẩn: đường thẳng từ cổ tay đến ngón giữa hướng thẳng lên trên (12 giờ),
· Sử dụng code để xoay ảnh và xác định góc xoay chính xác,
· Ghi nhãn “0” nếu giữ nguyên, không xoay, nhãn "loại" nếu loại bỏ, nhãn "hai tay" nếu có hai bàn tay không ghi số để tránh nhầm với số góc xoay.
1.4 QUY TẮC ĐÁNH NHÃN:
· File đánh nhãn phân công theo số thứ tự No. (Ví dụ từ No. 1 - No.500), không phân công theo theo ID ảnh,
· Đánh nhãn cần chính xác để mô hình học hiệu quả,
· Đánh nhãn loại với bàn tay dị tật, cụt ngón, thiếu ngón …
· Trường hợp bàn tay có các ngón tay bị cong ít thì vẫn đánh nhãn bình thường (vì nếu loại bỏ nhiều quá sẽ bị thiếu dữ liệu),
· Trường hợp bàn tay có các ngón tay bị co quắp nhiều sẽ ảnh hưởng đến việc học của mô hình nên thì đánh nhãn loại để loại bỏ hình này nhằm giúp mô hình học được hiệu quả hơn,
· Trường hợp phần xương ngón tay đè ít lên phần xương ngón tay khác, hoặc phần thịt ngón tay đè lên phần xương ngón tay khác thì vẫn đánh nhãn bình thường,
· Trường hợp phần xương ngón tay này đè nhiều lên phần xương ngón tay khác gây ảnh hưởng đến việc học của mô hình thì mới đánh nhãn loại,
· Trường hợp có 1 bàn tay ở trung tâm và có 1 phần bàn tay khác ở vị trí cách xa bàn tay ở trung tâm, không ảnh hưởng đến việc nhận diện bàn tay ở trung tâm thì vẫn đánh nhãn bình thường (không đánh nhãn loại),
· Trường hợp có 1 bàn tay ở trung tâm và có 1 phần bàn tay khác ở gần sát bàn tay ở trung tâm, gây ảnh hưởng đến việc nhận diện bàn tay ở trung tâm thì vẫn đánh loại.
1.5 CODE XOAY BÀN TAY ĐỂ KIỂM TRA GÓC XOAY:
Thực hiện đoạn code dưới đây: thay angle bằng số góc xoay đúng để xoay ảnh để hướng từ cổ tay đến ngón tay giữa là hướng lên trên (hướng 12 giờ). Sau khi tìm thấy số góc xoay đúng thì ghi vào file đánh nhãn. Đối với file giữ nguyên thì đánh nhãn số góc xoay là “0”, đối với file loại thì ghi nhãn “loại”, đối với file hai bàn tay thì ghi nhãn “hai tay” (không ghi số để tránh nhầm với số góc xoay):
from PIL import Image
img = Image.open(r"C:\Users\HP\OneDrive\Desktop\Penskull Education.png")
angle = 125
rotate_img= img.rotate(angle, expand = True)
rotate_img.show()
1.6 QUY TRÌNH UPLOAD FILE:
· Về việc upload file đánh nhãn (tay trái, phải, 2 tay, loại) vào thư mục: "Danh nhan anh tay trai phai 2024X Học máy (4) 4S", SV hoàn thành việc đánh nhãn thì đặt lại tên file theo số ID + Họ tên + Lớp sinh hoạt của mình và tải vào thư mục của nhóm mình. (Ví dụ 1: ID 1.1, Họ tên Nguyễn Văn An, Lớp sinh hoạt 21SE1 thì đặt tên file là TayTraiPhai2_1.1_NguyenVanAn_21SE1_tayall.xlsx và tải vào thư mục 1).
· Về việc upload file đánh nhãn (góc xoay) vào thư mục: "Danh nhan goc xoay 2024X Học máy (4) 4S", SV hoàn thành việc đánh nhãn thì đặt lại tên file theo số ID + Họ tên + Lớp sinh hoạt của mình và tải vào thư mục của nhóm mình. (Ví dụ 1: ID 1.1, Họ tên Nguyễn Văn An, Lớp sinh hoạt 21SE1 thì đặt tên file là GocXoay_1.1_NguyenVanAn_21SE1_tayall
Link thư mục thầy đã gửi cho các nhóm.
2 Công việc sau khi đánh nhãn:
· Sau khi đã đánh nhãn góc xoay. SV viết code để:
o Xoay ảnh hàng loạt file dựa trên nhãn góc xoay đã được SV đánh nhãn trước đó nhằm xoay các ảnh thành ảnh bàn tay tiêu chuẩn: đường thẳng từ cổ tay đến ngón giữa hướng thẳng lên trên (12 giờ),
o Lưu ảnh đã xoay trên 1 thư mục khác, không lưu đè lên ảnh gốc (vì nhãn góc xoay được đánh nhãn cho ảnh gốc, nếu lưu đè lên ảnh gốc thì nhãn góc xoay mà SV đã đánh nhãn sẽ không dùng được nữa do ảnh gốc đã bị thay đổi góc xoay).
· Bộ ảnh đã lưu sau khi xoay các file ảnh thành ảnh bàn tay tiêu chuẩn sẽ được dùng để huấn luyện,
· SV chọn thuật toán phù hợp để huấn luyện mô hình nhằm phân biệt bàn tay trái, bàn tay phải dựa trên:
o Phần dữ liệu đã giao riêng cho SV đánh nhãn.
o Phần dữ liệu đã giao riêng cho nhóm SV đánh nhãn.
o So sánh hiệu suất mô hình được huấn luyện dựa trên phần dữ liệu đã giao riêng cho SV đánh nhãn và phần dữ liệu giao riêng cho nhóm của SV đánh nhãn.