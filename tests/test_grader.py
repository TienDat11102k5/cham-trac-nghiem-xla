"""
Test cases cho module grader.py (TV4).

Chiến lược test:
  - Không dùng mock_data_generator.py
  - Mock data được tạo trực tiếp bằng numpy/cv2 trong từng test
  - Test thực tế dùng ảnh nan_chinh từ TV3 (nếu có)

Chạy:
    pytest tests/test_grader.py -v -s
hoặc:
    python tests/test_grader.py

Author: TV4
"""

import json
import sys
import pytest
import cv2
import numpy as np
from pathlib import Path

# Thêm thư mục gốc project vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grader import (
    extract_bubble_grid,
    segment_bubbles,
    calculate_score,
    export_result_json,
    print_result_table,
    grade_from_image,
)


# ============================================================
# HELPER: Tạo mock binary image cho test calculate_score
# ============================================================

def _make_binary_grid(answers: dict,
                      num_q: int = 4,
                      n_choices: int = 4,
                      bubble_h: int = 40,
                      bubble_w: int = 60) -> np.ndarray:
    """
    Tạo ảnh nhị phân giả lập lưới đáp án (layout tự động / chia đều).

    Mỗi hàng = 1 câu, mỗi cột = 1 lựa chọn (A B C D).
    Bubble được tô = vùng trắng giữa hàng/cột tương ứng.

    Args:
        answers (dict): {1: 'A', 2: 'C', ...}
        num_q: Số câu hỏi.
        n_choices: Số lựa chọn.
        bubble_h, bubble_w: Kích thước 1 ô.

    Returns:
        np.ndarray: Ảnh nhị phân shape (num_q*bubble_h, n_choices*bubble_w).
    """
    choices = ['A', 'B', 'C', 'D', 'E'][:n_choices]
    h = num_q * bubble_h
    w = n_choices * bubble_w
    binary = np.zeros((h, w), dtype=np.uint8)

    for q_num, chosen in answers.items():
        i = q_num - 1  # 0-indexed row
        if i >= num_q:
            continue
        j = choices.index(chosen) if chosen in choices else -1
        if j < 0:
            continue
        # Tô vùng trắng (padding 6px để giả sát thực tế)
        y1 = i * bubble_h + 6
        y2 = (i + 1) * bubble_h - 6
        x1 = j * bubble_w + 6
        x2 = (j + 1) * bubble_w - 6
        binary[y1:y2, x1:x2] = 255

    return binary


def _make_binary_with_erase(answers: dict,
                             erased_q: dict,
                             num_q: int = 4,
                             n_choices: int = 4,
                             bubble_h: int = 40,
                             bubble_w: int = 60) -> np.ndarray:
    """
    Tạo ảnh nhị phân có tẩy xóa bẩn.

    erased_q = {q_num: 'A'} → câu q_num có ô A bị tẩy còn mờ (intensity thấp).
    answers = đáp án mới sau khi tẩy và tô lại.

    Args:
        answers (dict): Đáp án mới (tô đậm).
        erased_q (dict): Đáp án cũ bị tẩy (tô mờ, ~35% white pixel).
    """
    binary = _make_binary_grid(answers, num_q, n_choices, bubble_h, bubble_w)
    choices = ['A', 'B', 'C', 'D'][:n_choices]

    for q_num, erased_choice in erased_q.items():
        i = q_num - 1
        j = choices.index(erased_choice) if erased_choice in choices else -1
        if j < 0:
            continue
        # Tô mờ vùng bị tẩy (35% pixel trắng = nhiễu tẩy xóa)
        y1 = i * bubble_h + 6
        y2 = (i + 1) * bubble_h - 6
        x1 = j * bubble_w + 6
        x2 = (j + 1) * bubble_w - 6
        noise_mask = np.random.rand(y2 - y1, x2 - x1) < 0.35
        binary[y1:y2, x1:x2] = np.where(noise_mask, 255, 0).astype(np.uint8)

    return binary


# ============================================================
# TEST CLASS 1: extract_bubble_grid
# ============================================================

class TestExtractBubbleGrid:
    """Test hàm extract_bubble_grid()"""

    def test_extract_roi_basic(self):
        """Cắt ROI cơ bản: shape output phải đúng."""
        img = np.zeros((1200, 800, 3), dtype=np.uint8)
        grid = extract_bubble_grid(img, roi_x=100, roi_y=200,
                                   roi_width=600, roi_height=800)
        assert grid.shape == (800, 600, 3), f"Shape sai: {grid.shape}"

    def test_extract_roi_grayscale(self):
        """Cắt ROI từ ảnh xám (2D)."""
        img = np.zeros((1200, 800), dtype=np.uint8)
        grid = extract_bubble_grid(img, roi_x=0, roi_y=0,
                                   roi_width=800, roi_height=1200)
        assert grid.shape == (1200, 800)

    def test_extract_roi_default_full_image(self):
        """roi_width=None, roi_height=None → lấy toàn bộ ảnh."""
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        grid = extract_bubble_grid(img, roi_x=50, roi_y=50)
        assert grid.shape == (450, 350, 3)

    def test_extract_roi_out_of_bounds_raises(self):
        """ROI vượt biên phải raise ValueError."""
        pytest.skip("Ham extract_bubble_grid da tu dong clip toa do, khong raise ValueError nua")

    def test_extract_roi_preserves_content(self):
        """Nội dung vùng cắt phải giữ nguyên giá trị pixel."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        grid = extract_bubble_grid(img, roi_x=10, roi_y=20,
                                   roi_width=50, roi_height=60)
        assert np.array_equal(grid, img[20:80, 10:60])


# ============================================================
# TEST CLASS 2: segment_bubbles
# ============================================================

class TestSegmentBubbles:
    """Test hàm segment_bubbles()"""

    @pytest.fixture
    def gray_with_dark_circle(self):
        """Ảnh xám 100×100 có 1 vòng tròn đen (bubble giả)."""
        img = np.full((100, 100), 220, dtype=np.uint8)
        cv2.circle(img, (50, 50), 15, 30, -1)  # vòng tròn đen
        return img

    def test_adaptive_output_binary(self, gray_with_dark_circle):
        """Output phải là ảnh nhị phân (chỉ 0 và 255)."""
        binary = segment_bubbles(gray_with_dark_circle, "adaptive")
        unique_vals = set(np.unique(binary).tolist())
        assert unique_vals.issubset({0, 255}), f"Có giá trị lạ: {unique_vals}"

    def test_adaptive_shape_preserved(self, gray_with_dark_circle):
        """Shape đầu ra phải bằng shape đầu vào."""
        binary = segment_bubbles(gray_with_dark_circle, "adaptive")
        assert binary.shape == gray_with_dark_circle.shape

    def test_adaptive_detects_dark_region(self, gray_with_dark_circle):
        """Vùng tối (bubble được tô) phải cho ra pixel trắng."""
        binary = segment_bubbles(gray_with_dark_circle, "adaptive")
        # Vùng tâm circle phải có pixel trắng (được tô)
        center_region = binary[35:65, 35:65]
        assert center_region.max() == 255, "Vùng tô phải có pixel trắng"

    def test_otsu_threshold(self, gray_with_dark_circle):
        """Otsu threshold phải trả về ảnh nhị phân hợp lệ."""
        binary = segment_bubbles(gray_with_dark_circle, "otsu")
        assert binary.ndim == 2
        assert binary.dtype == np.uint8
        assert set(np.unique(binary).tolist()).issubset({0, 255})

    def test_binary_threshold(self, gray_with_dark_circle):
        """Binary threshold phải trả về ảnh nhị phân hợp lệ."""
        binary = segment_bubbles(gray_with_dark_circle, "binary")
        assert binary.ndim == 2
        assert binary.dtype == np.uint8

    def test_color_input_auto_convert(self):
        """Ảnh màu đầu vào phải được tự động chuyển xám."""
        color_img = np.full((80, 240, 3), 200, dtype=np.uint8)
        cv2.circle(color_img, (40, 40), 15, (30, 30, 30), -1)
        binary = segment_bubbles(color_img, "adaptive")
        assert binary.ndim == 2, "Output phải là 2D"

    def test_invalid_method_raises(self):
        """Method không hợp lệ phải raise ValueError."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            segment_bubbles(img, "magic_threshold")
        assert "magic_threshold" in str(exc_info.value)


# ============================================================
# TEST CLASS 3: calculate_score (mock data)
# ============================================================

class TestCalculateScore:
    """Test hàm calculate_score() với mock binary image."""

    def test_all_correct(self):
        """Tất cả đáp án đúng → điểm 10.0."""
        answer_key = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        binary = _make_binary_grid(answer_key, num_q=4, n_choices=4)
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=4,
                                                  choices_per_question=4)
        assert correct == 4, f"Phải 4 câu đúng, được {correct}"
        assert score == 10.0, f"Phải điểm 10.0, được {score}"

    def test_partial_correct(self):
        """Một nửa đúng → điểm 5.0."""
        answer_key  = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        # Học sinh tô sai câu 3, 4
        student_ans = {1: 'A', 2: 'B', 3: 'A', 4: 'A'}
        binary = _make_binary_grid(student_ans, num_q=4, n_choices=4)
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=4,
                                                  choices_per_question=4)
        assert correct == 2, f"Phải 2 câu đúng, được {correct}"
        assert score == 5.0, f"Phải điểm 5.0, được {score}"

    def test_all_wrong(self):
        """Tất cả sai → điểm 0.0."""
        answer_key  = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        student_ans = {1: 'D', 2: 'C', 3: 'B', 4: 'A'}
        binary = _make_binary_grid(student_ans, num_q=4, n_choices=4)
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=4,
                                                  choices_per_question=4)
        assert correct == 0
        assert score == 0.0

    def test_blank_image_no_answer(self):
        """Ảnh toàn đen (không tô) → tất cả '?', 0 câu đúng."""
        answer_key = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        binary = np.zeros((160, 240), dtype=np.uint8)  # toàn đen
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=4,
                                                  choices_per_question=4)
        assert correct == 0
        for q in [1, 2, 3, 4]:
            assert answers[q] == '?', f"Câu {q} phải là '?', được '{answers[q]}'"

    def test_score_formula(self):
        """Điểm = (số_đúng / tổng_câu) × 10, làm tròn 2 chữ số."""
        answer_key  = {1: 'A', 2: 'B', 3: 'C'}
        student_ans = {1: 'A', 2: 'B', 3: 'A'}  # 2/3 đúng
        binary = _make_binary_grid(student_ans, num_q=3, n_choices=4)
        correct, score, _ = calculate_score(binary, answer_key,
                                            num_questions=3,
                                            choices_per_question=4)
        expected = round(2/3 * 10, 2)
        assert correct == 2
        assert score == expected, f"Phải {expected}, được {score}"

    def test_returns_correct_types(self):
        """Return types phải đúng: int, float, dict."""
        answer_key = {1: 'A', 2: 'B'}
        binary = _make_binary_grid(answer_key, num_q=2, n_choices=4)
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=2,
                                                  choices_per_question=4)
        assert isinstance(correct, int)
        assert isinstance(score, float)
        assert isinstance(answers, dict)


# ============================================================
# TEST CLASS 4: Tẩy xóa bẩn (dirty erase immunity)
# ============================================================

class TestEraseNoise:
    """Test khả năng miễn nhiễm tẩy xóa của z-score."""

    def test_erase_noise_still_correct(self):
        """
        Câu 1: học sinh tô A rồi tẩy, tô lại B → phải nhận ra B.
        Ô A sau tẩy có ~35% pixel trắng (nhiễu), ô B có ~80% pixel trắng.
        """
        answer_key  = {1: 'B', 2: 'C', 3: 'D', 4: 'A'}
        student_ans = {1: 'B', 2: 'C', 3: 'D', 4: 'A'}  # đáp án sau tẩy
        erased_q    = {1: 'A'}  # câu 1 tô A rồi tẩy

        binary = _make_binary_with_erase(student_ans, erased_q,
                                         num_q=4, n_choices=4,
                                         bubble_h=50, bubble_w=70)
        correct, score, answers = calculate_score(binary, answer_key,
                                                  num_questions=4,
                                                  choices_per_question=4)
        # Với z-score, câu 1 vẫn phải chọn B (đậm hơn nhiều)
        assert answers[1] == 'B', (
            f"Z-score phải chọn B (đáp án sau tẩy), được '{answers[1]}'"
        )
        assert correct >= 3, f"Phải đúng ít nhất 3/4 câu, được {correct}"


# ============================================================
# TEST CLASS 5: export_result_json
# ============================================================

class TestExportResultJson:
    """Test hàm export_result_json()"""

    def test_json_is_valid(self):
        """Output phải là JSON hợp lệ."""
        answer_key     = {1: 'A', 2: 'B'}
        student_answers = {1: 'A', 2: 'C'}
        json_str = export_result_json(1, 5.0, student_answers, answer_key)
        data = json.loads(json_str)  # không raise = hợp lệ
        assert isinstance(data, dict)

    def test_json_fields(self):
        """Các field bắt buộc phải có đủ."""
        answer_key     = {1: 'A', 2: 'B'}
        student_answers = {1: 'A', 2: 'C'}
        json_str = export_result_json(1, 5.0, student_answers, answer_key,
                                      image_path="test.jpg")
        data = json.loads(json_str)
        required = {"total", "correct", "wrong", "score", "score_display",
                    "details", "image_path"}
        assert required.issubset(data.keys()), \
            f"Thiếu fields: {required - data.keys()}"

    def test_json_score_values(self):
        """Giá trị điểm và số câu phải đúng."""
        answer_key     = {1: 'A', 2: 'B'}
        student_answers = {1: 'A', 2: 'C'}
        json_str = export_result_json(1, 5.0, student_answers, answer_key)
        data = json.loads(json_str)
        assert data["correct"]       == 1
        assert data["wrong"]         == 1
        assert data["total"]         == 2
        assert data["score"]         == 5.0
        assert data["score_display"] == "5.00/10"

    def test_json_details_result_field(self):
        """Details phải có field 'result': 'correct' hoặc 'wrong'."""
        answer_key     = {1: 'A', 2: 'B'}
        student_answers = {1: 'A', 2: 'C'}
        json_str = export_result_json(1, 5.0, student_answers, answer_key)
        data = json.loads(json_str)
        assert data["details"]["1"]["result"] == "correct"
        assert data["details"]["2"]["result"] == "wrong"

    def test_json_unicode_preserved(self):
        """Tiếng Việt và unicode không bị escape."""
        answer_key     = {1: 'A'}
        student_answers = {1: 'A'}
        json_str = export_result_json(1, 10.0, student_answers, answer_key,
                                      image_path="đề_thi_môn_vật_lý.jpg")
        assert "đề_thi_môn_vật_lý.jpg" in json_str


# ============================================================
# TEST CLASS 6: Test thực tế với ảnh từ TV3
# ============================================================

class TestWithRealImage:
    """
    Test tích hợp với ảnh thật từ TV3 (nếu có trong data/processed/).
    Bỏ qua tự động nếu không tìm thấy file.
    """

    NAN_CHINH_PATH = Path(__file__).parent.parent / \
        "data" / "processed" / "test_sheet_01_07_nan_chinh.jpg"
    ANSWER_KEY_PATH = Path(__file__).parent.parent / \
        "data" / "answer_keys" / "all_answer_keys.json"
    
    THU_MUC_NGUON = Path("data/raw")
    THU_MUC_DICH = Path("data/processed")

    @pytest.fixture
    def answer_key(self):
        """Load đáp án chuẩn từ JSON."""
        if not self.ANSWER_KEY_PATH.exists():
            pytest.skip(f"Không tìm thấy file đáp án: {self.ANSWER_KEY_PATH}")
        with open(self.ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Lấy mã đề đầu tiên
        first_exam = list(data.keys())[0]
        return {int(k): v for k, v in data[first_exam]["answers"].items()}

    def test_real_image_loads(self):
        """Ảnh nan_chinh phải đọc được và có shape đúng."""
        if not self.NAN_CHINH_PATH.exists():
            pytest.skip(f"Không tìm thấy: {self.NAN_CHINH_PATH}")
        img = cv2.imread(str(self.NAN_CHINH_PATH))
        assert img is not None
        h, w = img.shape[:2]
        assert h == 1200 and w == 800, f"Shape khác mong đợi: {img.shape}"

    def test_segment_real_image(self):
        """Phân đoạn ảnh thật phải trả về ảnh nhị phân hợp lệ."""
        if not self.NAN_CHINH_PATH.exists():
            pytest.skip(f"Không tìm thấy: {self.NAN_CHINH_PATH}")
        img  = cv2.imread(str(self.NAN_CHINH_PATH))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = segment_bubbles(gray, "adaptive")
        assert binary.ndim == 2
        assert binary.dtype == np.uint8
        assert binary.max() == 255, "Phải có pixel trắng (vùng được tô)"
        assert binary.min() == 0,   "Phải có pixel đen (nền)"

    def test_grade_real_image_returns_valid_result(self, answer_key):
        """
        Chạy toàn bộ pipeline với ảnh thật.
        Kiểm tra: correct_count hợp lệ, score trong [0,10], answers đủ 20 câu.
        """
        if not self.NAN_CHINH_PATH.exists():
            pytest.skip(f"Không tìm thấy: {self.NAN_CHINH_PATH}")

        img    = cv2.imread(str(self.NAN_CHINH_PATH))
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = segment_bubbles(gray, "adaptive")

        # Chỉ chấm 20 câu (ảnh test_sheet chỉ có 20 câu)
        key_20 = {k: v for k, v in answer_key.items() if k <= 20}

        # Không truyền tọa độ, để hàm tự động phát hiện layout
        correct, score, answers = calculate_score(
            binary,
            key_20,
            num_questions=20
        )

        # Assertions cơ bản
        assert 0 <= correct <= 20,      f"correct_count ngoài khoảng: {correct}"
        assert 0.0 <= score <= 10.0,    f"score ngoài khoảng: {score}"
        assert len(answers) == 20,      f"Phải có 20 câu, được {len(answers)}"

        # Tất cả đáp án phải là A/B/C/D hoặc '?'
        valid_choices = {'A', 'B', 'C', 'D', '?'}
        for q, ans in answers.items():
            assert ans in valid_choices, f"Câu {q}: đáp án không hợp lệ '{ans}'"

        # In kết quả để xem (chạy với -s)
        print(f"\n  [Real Image] correct={correct}/20, score={score}/10")
        print(f"  Student answers: {answers}")

    def test_grade_from_image_helper(self, answer_key):
        """Test hàm grade_from_image() — pipeline tất-cả-trong-một."""
        if not self.NAN_CHINH_PATH.exists():
            pytest.skip(f"Không tìm thấy: {self.NAN_CHINH_PATH}")

        img = cv2.imread(str(self.NAN_CHINH_PATH))
        key_20 = {k: v for k, v in answer_key.items() if k <= 20}
        correct, score, answers = grade_from_image(
            img, key_20
        )
        assert isinstance(correct, int)
        assert isinstance(score, float)
        assert len(answers) == 20
    
    def test_export_grading_images(self, answer_key):
        """Test xuất ảnh vùng đáp án và ảnh binary để debug"""
        from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
        from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
        from src.reader import extract_answer_region
        
        danh_sach_anh = list(self.THU_MUC_NGUON.glob("*.jpg"))
        if not danh_sach_anh:
            pytest.skip("Không có ảnh trong data/raw/")
        
        self.THU_MUC_DICH.mkdir(parents=True, exist_ok=True)
        ket_qua = []
        
        for duong_dan_anh in danh_sach_anh:
            ten_goc = duong_dan_anh.stem
            
            try:
                # Bước 1: Đọc và tiền xử lý
                anh_goc = doc_anh(str(duong_dan_anh))
                anh_xam = chuyen_xam(anh_goc)
                anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
                anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
                
                # Bước 2: Tìm góc và nắn chỉnh
                cac_goc = tim_goc_giay(anh_canh)
                if cac_goc is None:
                    print(f"Bỏ qua {ten_goc}: Không tìm thấy 4 góc")
                    continue
                
                anh_thang = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
                
                # Bước 3: Xuất ảnh nắn chỉnh
                duong_dan_thang = self.THU_MUC_DICH / f"{ten_goc}_07_nan_chinh.jpg"
                cv2.imwrite(str(duong_dan_thang), anh_thang)
                ket_qua.append(duong_dan_thang.name)
                
                # Bước 4: Cắt vùng đáp án
                dap_an_region = extract_answer_region(anh_thang)
                duong_dan_dap_an = self.THU_MUC_DICH / f"{ten_goc}_dap_an_region.jpg"
                cv2.imwrite(str(duong_dan_dap_an), dap_an_region)
                ket_qua.append(duong_dan_dap_an.name)
                
                # Bước 5: Segment bubbles (binary)
                binary = segment_bubbles(dap_an_region, "adaptive")
                duong_dan_binary = self.THU_MUC_DICH / f"{ten_goc}_dap_an_binary.jpg"
                cv2.imwrite(str(duong_dan_binary), binary)
                ket_qua.append(duong_dan_binary.name)
                
            except Exception as e:
                print(f"Lỗi xử lý {ten_goc}: {e}")
                continue
        
        assert len(ket_qua) > 0, "Không xuất được ảnh nào"
        print(f"\nĐã xuất {len(ket_qua)} ảnh vào {self.THU_MUC_DICH}:")
        for ten in ket_qua:
            print(f"  - {ten}")


# ============================================================
# CHẠY TRỰC TIẾP (không dùng pytest)
# ============================================================

def _run_all_manual():
    """Chạy thủ công khi không có pytest."""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"

    def run(name, fn):
        try:
            fn()
            print(f"{PASS}  {name}")
            return True
        except Exception as e:
            print(f"{FAIL}  {name}  [{type(e).__name__}: {e}]")
            return False

    tests = [
        # extract_bubble_grid
        ("extract_roi_basic",          lambda: TestExtractBubbleGrid().test_extract_roi_basic()),
        ("extract_roi_grayscale",      lambda: TestExtractBubbleGrid().test_extract_roi_grayscale()),
        ("extract_roi_full_image",     lambda: TestExtractBubbleGrid().test_extract_roi_default_full_image()),
        ("extract_roi_out_of_bounds",  lambda: TestExtractBubbleGrid().test_extract_roi_out_of_bounds_raises()),
        ("extract_roi_content",        lambda: TestExtractBubbleGrid().test_extract_roi_preserves_content()),

        # segment_bubbles — cần fixture nên khởi tạo trực tiếp
        ("segment_adaptive_binary",    lambda: _test_segment_helper("adaptive")),
        ("segment_otsu_binary",        lambda: _test_segment_helper("otsu")),
        ("segment_binary_threshold",   lambda: _test_segment_helper("binary")),
        ("segment_color_input",        lambda: TestSegmentBubbles().test_color_input_auto_convert()),

        # calculate_score
        ("score_all_correct",          lambda: TestCalculateScore().test_all_correct()),
        ("score_partial",              lambda: TestCalculateScore().test_partial_correct()),
        ("score_all_wrong",            lambda: TestCalculateScore().test_all_wrong()),
        ("score_blank_image",          lambda: TestCalculateScore().test_blank_image_no_answer()),
        ("score_formula",              lambda: TestCalculateScore().test_score_formula()),
        ("score_return_types",         lambda: TestCalculateScore().test_returns_correct_types()),

        # erase noise
        ("erase_noise_immunity",       lambda: TestEraseNoise().test_erase_noise_still_correct()),

        # export json
        ("json_valid",                 lambda: TestExportResultJson().test_json_is_valid()),
        ("json_fields",                lambda: TestExportResultJson().test_json_fields()),
        ("json_score_values",          lambda: TestExportResultJson().test_json_score_values()),
        ("json_details_result",        lambda: TestExportResultJson().test_json_details_result_field()),
        ("json_unicode",               lambda: TestExportResultJson().test_json_unicode_preserved()),
    ]

    print("\n" + "=" * 58)
    print("   TEST MODULE grader.py — TV4")
    print("=" * 58)

    passed = sum(run(name, fn) for name, fn in tests)
    total  = len(tests)

    print("=" * 58)
    print(f"   Kết quả: {passed}/{total} test passed")
    print("=" * 58)

    if passed == total:
        print("🎉 Tất cả test PASS! grader.py sẵn sàng tích hợp.\n")
    else:
        print("⚠️  Có test FAIL. Kiểm tra lại trước khi tích hợp.\n")


def _test_segment_helper(method: str):
    """Helper cho segment test không dùng fixture."""
    img = np.full((100, 100), 220, dtype=np.uint8)
    cv2.circle(img, (50, 50), 15, 30, -1)
    binary = segment_bubbles(img, method)
    assert binary.ndim == 2
    assert binary.dtype == np.uint8
    assert set(np.unique(binary).tolist()).issubset({0, 255})


if __name__ == "__main__":
    _run_all_manual()