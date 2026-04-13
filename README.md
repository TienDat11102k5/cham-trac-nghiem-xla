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

```bash
git clone <repository-url>
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

- [ ] **Module preprocessing.py**: Đọc ảnh, chuyển xám, khử nhiễu
- [ ] **Module transform.py**: Phát hiện biên, tìm góc, nắn chỉnh
- [ ] **Module grader.py**: Trích xuất ROI, phân ngưỡng, chấm điểm
- [ ] **Module utils.py**: Các hàm tiện ích (hiển thị, lưu ảnh, vẽ contours)
- [ ] **File main.py**: Hoàn thiện pipeline chính
- [ ] **Notebook demo**: Tạo file Jupyter minh họa từng bước

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
