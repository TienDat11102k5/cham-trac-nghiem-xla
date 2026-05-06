# BẢNG PHÂN CÔNG THÀNH VIÊN NHÓM

## Đồ án: Hệ thống Chấm Trắc Nghiệm Tự Động (OMR)

---

## Bảng Phân Công Chi Tiết

| STT | Họ và Tên | Vai trò | Thành phần phụ trách | Mô tả công việc |
|-----|-----------|---------|---------------------|-----------------|
| 1 | **Đỗ Tiến Đạt** | Lập trình viên Cốt lõi (Reader) | Module `src/reader.py` | Phát hiện anchor markers, phân loại vùng ROI (SBD, Mã đề, Đáp án), trích xuất vùng đáp án, đọc mã đề và SBD bằng HoughCircles + Z-score |
| 2 | **Phạm Võ Thành Đạt** | Lập trình viên Cốt lõi 1 (Preprocessing) | Module `src/preprocessing.py` | Đọc ảnh, chuyển xám, lọc nhiễu (Gaussian/Median/Bilateral), hỗ trợ TV1 phát hiện cạnh Canny |
| 3 | **Lưu Nhất Huy** | Lập trình viên Cốt lõi 2 (Transform) | Module `src/transform.py` | Phát hiện biên Canny, tìm 4 góc tờ giấy, áp dụng Perspective Transform nắn chỉnh ảnh |
| 4 | **Nguyễn Thị Thu Vân** | Lập trình viên Cốt lõi 3 (Grader) | Module `src/grader.py` | Phân đoạn bubble, phát hiện lưới đáp án, chấm điểm, xuất kết quả JSON |
| 5 | **Nguyễn An Kha** | Chuyên viên Dữ liệu & QA | Module `tests/` (5 files test) | Viết test cases (>100 cases), kiểm thử tự động, tạo Ground Truth, đánh giá Accuracy |
| 6 | **Nguyễn Gia Quy** | Tools Debug & Báo cáo | Module `tools/` (6 files Python) + Tài liệu | Phát triển tools debug/automation, Jupyter Notebook demo, tài liệu kỹ thuật, báo cáo khoa học |

