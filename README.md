# 📝 Hệ thống Chấm Trắc Nghiệm Tự Động (OMR)

Đồ án môn **Xử lý Ảnh** - Optical Mark Recognition System sử dụng Python và OpenCV.

## Mô tả

Hệ thống tự động chấm điểm bài thi trắc nghiệm từ ảnh chụp/scan, sử dụng các kỹ thuật Computer Vision:

- Tiền xử lý ảnh (Grayscale, Noise Filtering)
- Phát hiện biên (Canny Edge Detection)
- Biến đổi phối cảnh (Perspective Transform)
- Phân ngưỡng (Thresholding)
- Phân tích vùng đáp án và chấm điểm

## Cấu trúc dự án

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
│   ├── reader.py               # Module đọc thông tin (mã đề, MSSV)
│   ├── grader.py               # Module chấm điểm
│   └── utils.py                # Các hàm tiện ích
│
├── .gitignore                  # File chặn đẩy rác lên Git
├── main.py                     # File chính chạy pipeline
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## Cài đặt

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

## Hướng dẫn sử dụng

### Chạy pipeline chấm điểm

```bash
python main.py data/raw/test_sheet_01.jpg data/answer_key.json
```

### Chạy Jupyter Notebook demo

```bash
jupyter notebook notebooks/pipeline_demo.ipynb
```

## Pipeline xử lý

1. **Đọc ảnh** (`preprocessing.load_image`)
2. **Chuyển sang ảnh xám** (`preprocessing.convert_to_grayscale`)
3. **Khử nhiễu** (`preprocessing.apply_noise_filter`)
4. **Phát hiện biên** (`transform.detect_edges`)
5. **Tìm 4 góc tờ giấy** (`transform.find_document_corners`)
6. **Nắn chỉnh ảnh** (`transform.apply_perspective_transform`)
7. **Đọc mã đề** (`reader.read_exam_code`) - Tùy chọn
8. **Trích xuất vùng đáp án** (`grader.extract_bubble_grid`)
9. **Phân đoạn ô trắc nghiệm** (`grader.segment_bubbles`)
10. **Chấm điểm** (`grader.calculate_score`)

## Format đáp án chuẩn (JSON)

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

## Nhiệm vụ phân công

### Thành viên 1: Đỗ Tiến Đạt - Lập trình viên Cốt lõi (Module Reader)

- [X] Thiết lập Repository, viết khung main.py
- [X] Định nghĩa các tham số (Interface/API) đầu vào/ra cho các Core Dev
- [X] Phân nhánh Github, đóng gói mã nguồn & viết README
- [X] Lập trình module `reader.py` (Phát hiện anchor, đọc mã đề, SBD)
- [X] Phát hiện anchor markers bằng contour detection
- [X] Phân loại vùng ROI (SBD, Mã đề, Đáp án)
- [X] Đọc mã đề và số báo danh bằng HoughCircles + Z-score
- [X] Viết test cases cho module reader

### Thành viên 2: Phạm Võ Thành Đạt - Lập trình viên Cốt lõi 1 (Tiền xử lý & Hỗ trợ Reader)

- [X] Khởi tạo hàm đọc ảnh
- [X] Lập trình bộ lọc nhiễu (Gaussian, Median, Bilateral)
- [X] Hỗ trợ TV1 phát hiện cạnh (Canny) cho module reader
- [X] Hỗ trợ TV1 tối ưu thuật toán phát hiện anchor
- [X] Viết test cases cho module preprocessing

### Thành viên 3: Lưu Nhất Huy - Lập trình viên Cốt lõi 2 (Biến đổi hình học)

- [X] Nhận ảnh từ TV2, lập trình thuật toán tìm 4 góc
- [X] Áp dụng Biến đổi phối cảnh (Perspective Transform) nắn phẳng tờ giấy bị nghiêng
- [X] Đóng gói thành module độc lập (src/transform.py)
- [X] Tối ưu chống cắt lẹm góc/nhầm khung (sử dụng morphology)
- [X] Viết test cases đầy đủ

### Thành viên 4: Nguyễn Thị Thu Vân - Lập trình viên Cốt lõi 3 (Phân đoạn & Chấm điểm)

- [X] Không đợi TV3: Tự chuẩn bị ảnh mẫu (mock data)
- [X] Cắt tọa độ lưới đáp án
- [X] Phân ngưỡng mức đen & Đếm logic tính điểm
- [X] Viết test cases với mock data
- [X] Tích hợp module grader.py với pipeline

### Thành viên 5: Nguyễn An Khả - Chuyên viên Dữ liệu & Kiểm thử QA

- [ ] Thu thập Đợt 2 (Tập Test bí mật có ảnh khó)
- [ ] Xây dựng kịch bản kiểm thử (Test cases)
- [ ] Chạy test tự động và báo Bug cho Dev
- [ ] Tổng hợp số liệu đo kiểm thử, đánh giá Accuracy
- [ ] Tạo nhãn Ground Truth cho tập dữ liệu

### Thành viên 6: Nguyễn Gia Quý - Tools Debug và Chuyên viên viết báo cáo

- [X] Phát triển module tools debug pipeline, batch processing, phân tích anchor
- [X] Xây dựng Jupyter Notebook demo: `notebooks/pipeline_demo.ipynb`
- [X] Viết tài liệu kỹ thuật: `docs/BaoCaoDoAn.md`
- [X] Hoàn thiện báo cáo khoa học

## Công nghệ sử dụng

- **Python 3.8+**
- **OpenCV 4.8+**: Xử lý ảnh
- **NumPy**: Tính toán ma trận
- **Jupyter Notebook**: Demo và visualization

## Thành viên nhóm

| STT | Họ và Tên | Vai trò | Nhiệm vụ chính |
|-----|-----------|---------|----------------|
| 1 | **Đỗ Tiến Đạt** | Lập trình viên Cốt lõi (Reader) | Module reader.py, phát hiện anchor, đọc mã đề/SBD |
| 2 | **Phạm Võ Thành Đạt** | Lập trình viên Cốt lõi 1 (Preprocessing) | Module preprocessing.py, hỗ trợ TV1 |
| 3 | **Lưu Nhất Huy** | Lập trình viên Cốt lõi 2 (Transform) | Module transform.py, perspective transform |
| 4 | **Nguyễn Thị Thu Vân** | Lập trình viên Cốt lõi 3 (Grader) | Module grader.py, chấm điểm |
| 5 | **Nguyễn An Khả** | Chuyên viên Dữ liệu & QA | Kiểm thử, tạo Ground Truth, đánh giá Accuracy |
| 6 | **Nguyễn Gia Quý** | Tools Debug & Báo cáo | Module tools/, Jupyter Notebook, tài liệu kỹ thuật, báo cáo khoa học |

## Tài liệu tham khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Perspective Transform](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Adaptive Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

## License

Dự án này được phát triển cho mục đích học tập.

---

**Chúc các bạn code vui vẻ! 🎉**
