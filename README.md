# 📝 Hệ thống Chấm Trắc Nghiệm Tự Động (OMR)

Đồ án môn **Xử lý Ảnh** - Optical Mark Recognition System sử dụng Python và OpenCV.

## 📋 Mô tả

Hệ thống tự động chấm điểm bài thi trắc nghiệm từ ảnh chụp/scan, sử dụng các kỹ thuật Computer Vision:

- Tiền xử lý ảnh (Grayscale, Noise Filtering)
- Phát hiện biên (Canny Edge Detection)
- Biến đổi phối cảnh (Perspective Transform)
- Phân ngưỡng (Thresholding)
- Phân tích vùng đáp án và chấm điểm

## 🏗️ Cấu trúc dự án

```
cham-trac-nghiem-xla/
│
├── data/
│   ├── raw/                    # Ảnh gốc (Không push lên Git)
│   └── processed/              # Ảnh đã xử lý
│
├── notebooks/
│   └── pipeline_demo.ipynb     # File Jupyter demo từng bước xử lý
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Cấu hình hệ thống (hằng số, ngưỡng, kích thước)
│   ├── preprocessing.py        # Module tiền xử lý ảnh
│   ├── transform.py            # Module biến đổi hình học
│   ├── grader.py               # Module chấm điểm
│   └── utils.py                # Các hàm tiện ích
│
├── .gitignore                  # File chặn đẩy rác lên Git
├── main.py                     # File chính chạy pipeline
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## 🚀 Cài đặt

### 1. Clone repository

**Lưu ý:** Repository này là private. Các thành viên trong nhóm cần được thêm vào Collaborators:

- Vào GitHub repo → Settings → Collaborators → Add people
- Thêm GitHub username của từng thành viên

```bash
git clone https://github.com/TienDat11102k5/cham-trac-nghiem-xla.git
cd cham-trac-nghiem-xla
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### Chạy pipeline chấm điểm

```bash
python main.py data/raw/test_sheet_01.jpg data/answer_key.json
```

### Chạy Jupyter Notebook demo

```bash
jupyter notebook notebooks/pipeline_demo.ipynb
```

## 🔧 Pipeline xử lý

1. **Đọc ảnh** (`preprocessing.load_image`)
2. **Chuyển sang ảnh xám** (`preprocessing.convert_to_grayscale`)
3. **Khử nhiễu** (`preprocessing.apply_noise_filter`)
4. **Phát hiện biên** (`transform.detect_edges`)
5. **Tìm 4 góc tờ giấy** (`transform.find_document_corners`)
6. **Nắn chỉnh ảnh** (`transform.apply_perspective_transform`)
7. **Trích xuất vùng đáp án** (`grader.extract_bubble_grid`)
8. **Phân đoạn ô trắc nghiệm** (`grader.segment_bubbles`)
9. **Chấm điểm** (`grader.calculate_score`)

## 📝 Format đáp án chuẩn (JSON)

Tạo file `data/answer_key.json`:

```json
{
    "1": "A",
    "2": "C",
    "3": "B",
    "4": "D",
    "5": "A",
    ...
}
```

## 🎯 Nhiệm vụ phân công

### Thành viên 1: Quản trị Hệ thống & Kiến trúc

- [X] Thiết lập Repository, viết khung main.py
- [X] Định nghĩa các tham số (Interface/API) đầu vào/ra cho các Core Dev
- [X] Phân nhánh Github, đóng gói mã nguồn & viết README

### Thành viên 2: Lập trình viên Cốt lõi 1 (Tiền xử lý)

- [X] Khởi tạo hàm đọc ảnh
- [X] Lập trình viên Cốt lõi 1 (Gaussian, Median...) để làm sạch nhiễu
- [X] Hỗ trợ TV3 tìm cạnh (Canny)
- [X] Viết test cases cho module preprocessing

### Thành viên 3: Lập trình viên Cốt lõi 2 (Biến đổi hình học)

- [ ] Nhận ảnh từ TV2, lập trình lấy thuật toán tìm 4 góc
- [ ] Áp dụng Biến đổi phối cảnh (Perspective Transform) nắn phẳng tờ giấy bị nghiêng
- [ ] Viết test cases cho module transform

### Thành viên 4: Lập trình viên Cốt lõi 3 (Phân đoạn & Chấm điểm)

- [ ] Không đợi TV3: Tự chuẩn bị ảnh mẫu (mock data)
- [ ] Cắt tọa độ lưới đáp án
- [ ] Phân ngưỡng mức đen & Đếm logic tính điểm
- [ ] Viết test cases với mock data

### Thành viên 5: Chuyên viên Dữ liệu & Kiểm thử QA

- [ ] Thu thập Đợt 2 (Tập Test bí mật có ảnh khó)
- [ ] Xây dựng kịch bản kiểm thử (Test cases)
- [ ] Chạy test tự động và báo Bug cho Dev
- [ ] Tổng hợp số liệu đo kiểm thử, đánh giá Accuracy

### Thành viên 6: Chuyên trách Báo cáo khoa học

- [ ] Tổng hợp tài liệu/lý thuyết
- [ ] Lên khung mục lục
- [ ] Phụ giúp TV5 tạo nhãn Ground Truth và quyết định các thang đo (Accuracy...)
- [ ] Phỏng vấn Dev để viết sơ đồ lưu thuật toán
- [ ] Tổng hợp số liệu do kiểm thử, chỉnh xác từ TV5 vào chương kết quả
- [ ] Hoàn thiện format, lưu giữ dây dư lý do chọn thuật toán

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**
- **OpenCV 4.8+**: Xử lý ảnh
- **NumPy**: Tính toán ma trận
- **Jupyter Notebook**: Demo và visualization

## 👥 Thành viên nhóm

- Thành viên 1: [Tên] - [Nhiệm vụ]
- Thành viên 2: [Tên] - [Nhiệm vụ]
- Thành viên 3: [Tên] - [Nhiệm vụ]

## 📚 Tài liệu tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Perspective Transform](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Adaptive Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

## 📄 License

Dự án này được phát triển cho mục đích học tập.

---

**Chúc các bạn code vui vẻ! 🎉**
