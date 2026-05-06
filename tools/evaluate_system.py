"""
Script đánh giá hiệu năng hệ thống OMR trên toàn bộ tập test.

Tính toán các chỉ số:
- Accuracy đọc đáp án (so với đáp án học sinh đã tô)
- Accuracy đọc mã đề
- Accuracy đọc SBD
- Tỉ lệ detect anchor thành công
- Tỉ lệ perspective transform thành công
- Thời gian xử lý trung bình
"""

import sys
import json
import time
import csv
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Unicode output trên Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
from src.reader import (
    phat_hien_anchor, phan_loai_vung_roi,
    read_exam_code, read_student_id,
    extract_exam_code_region, extract_student_id_region
)
from src.grader import grade_from_image


def load_ground_truth(answer_keys_dir: Path) -> Dict[str, Dict]:
    """Đọc tất cả ground truth từ thư mục answer_keys."""
    ground_truth = {}
    for json_file in sorted(answer_keys_dir.glob("result_test_sheet_*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract sheet number from filename: result_test_sheet_01 -> "01"
        sheet_num = json_file.stem.replace("result_test_sheet_", "")
        ground_truth[sheet_num] = data
    return ground_truth


def evaluate_single_image(image_path: Path, ground_truth: Dict, all_answer_keys: Dict) -> Dict:
    """Đánh giá 1 ảnh và trả về kết quả chi tiết."""
    result = {
        'image': image_path.name,
        'success': False,
        'anchor_detected': False,
        'transform_success': False,
        'exam_code_correct': False,
        'exam_code_predicted': None,
        'exam_code_gt': ground_truth.get('ma_de', None),
        'student_id_correct': False,
        'student_id_predicted': None,
        'student_id_gt': ground_truth.get('so_bao_danh', None),
        'answers_correct': 0,
        'answers_total': 0,
        'answers_wrong_list': [],   # danh sách câu đọc sai
        'processing_time': 0,
        'errors': []
    }

    start_time = time.time()

    try:
        # ── Bước 1: Đọc và tiền xử lý ──
        anh = doc_anh(str(image_path))
        anh_xam = chuyen_xam(anh)
        anh_loc = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)

        # ── Bước 2: Perspective Transform (copy đúng pipeline main.py) ──
        anh_canh = tim_canh(anh_loc, nguong_thap=50, nguong_cao=150)
        cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=True)

        if cac_goc is not None:
            result['transform_success'] = True
            anh_nan = nan_chinh_anh(anh, cac_goc, chieu_rong=800, chieu_cao=1200)
        else:
            anh_nan = cv2.resize(anh, (800, 1200))

        # ── Bước 3: Phát hiện anchor ──
        anchors = []
        rois = None
        try:
            anchors = phat_hien_anchor(anh_nan)
            result['anchor_detected'] = True
            rois = phan_loai_vung_roi(anchors, *anh_nan.shape[:2])
        except Exception as e:
            result['errors'].append(f"Anchor detection failed: {str(e)}")

        # ── Bước 4: Đọc mã đề ──
        ma_de_gt = ground_truth.get('ma_de', None)
        ma_de_pred = None
        try:
            if rois and 'ma_de' in rois:
                ma_de_region = extract_exam_code_region(anh_nan, *rois['ma_de'])
            else:
                ma_de_region = extract_exam_code_region(anh_nan)
                
            for method in ["otsu", "adaptive", "binary"]:
                try:
                    ma_de_pred = read_exam_code(ma_de_region, num_digits=3, threshold_method=method)
                    break
                except ValueError:
                    pass
            
            result['exam_code_predicted'] = ma_de_pred
            if ma_de_gt is not None and ma_de_pred is not None:
                result['exam_code_correct'] = (str(ma_de_pred) == str(ma_de_gt))
        except Exception as e:
            result['errors'].append(f"Exam code reading failed: {str(e)}")

        # ── Bước 5: Đọc SBD ──
        sbd_gt = ground_truth.get('so_bao_danh', None)
        sbd_pred = None
        try:
            if rois and 'sbd' in rois:
                sbd_region = extract_student_id_region(anh_nan, *rois['sbd'])
            else:
                sbd_region = extract_student_id_region(anh_nan)
                
            num_digits = len(str(sbd_gt)) if sbd_gt else 6
            for method in ["otsu", "adaptive", "binary"]:
                try:
                    sbd_pred = read_student_id(sbd_region, num_digits=num_digits, threshold_method=method)
                    break
                except ValueError:
                    pass
            
            result['student_id_predicted'] = sbd_pred
            if sbd_gt is not None and sbd_pred is not None:
                result['student_id_correct'] = (str(sbd_pred) == str(sbd_gt))
        except Exception as e:
            result['errors'].append(f"Student ID reading failed: {str(e)}")

        # ── Bước 6: Chấm điểm đáp án ──
        # Chọn đáp án đúng theo mã đề (dùng ma_de_pred nếu có, fallback gt)
        exam_code_to_use = result.get('exam_code_predicted') or str(ma_de_gt or '101')

        answer_key_raw = None
        if exam_code_to_use in all_answer_keys:
            answer_key_raw = all_answer_keys[exam_code_to_use]['answers']
        else:
            # Thử fallback về mã đề ground truth
            if str(ma_de_gt) in all_answer_keys:
                answer_key_raw = all_answer_keys[str(ma_de_gt)]['answers']
            else:
                # Lấy key đầu tiên
                first_key = list(all_answer_keys.keys())[0]
                answer_key_raw = all_answer_keys[first_key]['answers']

        answer_key_int = {int(k): v for k, v in answer_key_raw.items()}

        try:
            correct_count, score, student_answers = grade_from_image(
                anh_nan,
                answer_key_int,
                num_questions=20
            )

            # So sánh đáp án đọc được với đáp án học sinh ĐÃ TÔ (ground truth student)
            gt_details = ground_truth.get('details', {})
            gt_answers = ground_truth.get('answers', {})

            for q_num in range(1, 21):
                q_str = str(q_num)

                # Lấy đáp án ground truth của học sinh (không phải đáp án đúng)
                if q_str in gt_details:
                    gt_student_ans = gt_details[q_str]['student']
                elif q_str in gt_answers:
                    gt_student_ans = gt_answers[q_str]
                else:
                    continue

                # Đáp án hệ thống đọc được
                sys_ans = student_answers.get(q_num, '?')

                result['answers_total'] += 1
                if sys_ans == gt_student_ans:
                    result['answers_correct'] += 1
                else:
                    result['answers_wrong_list'].append(
                        f"Q{q_num}: got={sys_ans}, gt={gt_student_ans}"
                    )

            result['success'] = True

        except Exception as e:
            result['errors'].append(f"Grading failed: {str(e)}")

    except Exception as e:
        result['errors'].append(f"Pipeline failed: {str(e)}")

    result['processing_time'] = time.time() - start_time
    return result


def calculate_metrics(results: List[Dict]) -> Dict:
    """Tính toán các chỉ số tổng hợp."""
    total_images = len(results)

    anchor_success   = sum(1 for r in results if r['anchor_detected'])
    transform_success= sum(1 for r in results if r['transform_success'])
    exam_code_correct= sum(1 for r in results if r['exam_code_correct'])
    student_id_correct=sum(1 for r in results if r['student_id_correct'])
    pipeline_success = sum(1 for r in results if r['success'])

    total_answers  = sum(r['answers_total']   for r in results)
    correct_answers= sum(r['answers_correct'] for r in results)

    times = [r['processing_time'] for r in results]
    avg_time = sum(times) / total_images if total_images > 0 else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0

    metrics = {
        'total_images'            : total_images,
        'pipeline_success_count'  : pipeline_success,
        'anchor_success_count'    : anchor_success,
        'transform_success_count' : transform_success,
        'exam_code_correct_count' : exam_code_correct,
        'student_id_correct_count': student_id_correct,
        'correct_answers'         : correct_answers,
        'total_answers'           : total_answers,

        'pipeline_success_rate'   : (pipeline_success   / total_images * 100) if total_images > 0 else 0,
        'anchor_detection_rate'   : (anchor_success     / total_images * 100) if total_images > 0 else 0,
        'transform_success_rate'  : (transform_success  / total_images * 100) if total_images > 0 else 0,
        'exam_code_accuracy'      : (exam_code_correct  / total_images * 100) if total_images > 0 else 0,
        'student_id_accuracy'     : (student_id_correct / total_images * 100) if total_images > 0 else 0,
        'answer_accuracy'         : (correct_answers    / total_answers * 100) if total_answers > 0 else 0,

        'avg_processing_time'     : avg_time,
        'min_processing_time'     : min_time,
        'max_processing_time'     : max_time,
    }
    return metrics


def count_pytest_tests(base_dir: Path) -> int:
    """Đếm tổng số test cases trong thư mục tests/."""
    import re
    tests_dir = base_dir / "tests"
    total = 0
    if tests_dir.exists():
        for f in tests_dir.rglob("test_*.py"):
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                # Đếm tất cả hàm def test_... (kể cả trong class)
                total += len(re.findall(r'^\s*def test_', content, re.MULTILINE))
            except Exception:
                pass
    return total


def count_test_code_lines(base_dir: Path) -> int:
    """Đếm số dòng code trong thư mục tests/."""
    tests_dir = base_dir / "tests"
    total = 0
    if tests_dir.exists():
        for f in tests_dir.rglob("*.py"):
            try:
                lines = f.read_text(encoding='utf-8', errors='ignore').splitlines()
                total += len([l for l in lines if l.strip()])
            except Exception:
                pass
    return total


def export_csv(results: List[Dict], output_path: Path):
    """Xuất kết quả chi tiết từng ảnh ra CSV."""
    fieldnames = [
        'image', 'success', 'anchor_detected', 'transform_success',
        'exam_code_gt', 'exam_code_predicted', 'exam_code_correct',
        'student_id_gt', 'student_id_predicted', 'student_id_correct',
        'answers_correct', 'answers_total', 'answer_accuracy_pct',
        'processing_time', 'errors'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            acc = (r['answers_correct'] / r['answers_total'] * 100) if r['answers_total'] > 0 else 0
            writer.writerow({
                'image'                : r['image'],
                'success'              : r['success'],
                'anchor_detected'      : r['anchor_detected'],
                'transform_success'    : r['transform_success'],
                'exam_code_gt'         : r['exam_code_gt'],
                'exam_code_predicted'  : r['exam_code_predicted'],
                'exam_code_correct'    : r['exam_code_correct'],
                'student_id_gt'        : r['student_id_gt'],
                'student_id_predicted' : r['student_id_predicted'],
                'student_id_correct'   : r['student_id_correct'],
                'answers_correct'      : r['answers_correct'],
                'answers_total'        : r['answers_total'],
                'answer_accuracy_pct'  : f"{acc:.1f}",
                'processing_time'      : f"{r['processing_time']:.2f}",
                'errors'               : '; '.join(r['errors'])
            })


def main():
    """Chạy đánh giá trên toàn bộ tập test."""
    print("=" * 80)
    print("  ĐÁNH GIÁ HIỆU NĂNG HỆ THỐNG OMR — BẢNG ĐỊNH LƯỢNG ĐỒ ÁN")
    print("=" * 80)
    print()

    base_dir       = Path(__file__).parent.parent
    raw_dir        = base_dir / "data" / "raw"
    answer_keys_dir= base_dir / "data" / "answer_keys"
    output_dir     = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # ── Load ground truth ──
    print("📂 Đang đọc Ground Truth...")
    ground_truth = load_ground_truth(answer_keys_dir)
    print(f"   ✓ Đã load {len(ground_truth)} file ground truth")

    # ── Load answer keys ──
    answer_keys_path = answer_keys_dir / "all_answer_keys.json"
    with open(answer_keys_path, 'r', encoding='utf-8') as f:
        all_answer_keys = json.load(f)
    print(f"   ✓ Đã load {len(all_answer_keys)} mã đề")
    print()

    # ── Lấy danh sách ảnh ──
    image_files = sorted(raw_dir.glob("test_sheet_*.jpg"))
    print(f"📸 Tìm thấy {len(image_files)} ảnh test: {[f.name for f in image_files]}")
    print()

    # ── Đánh giá từng ảnh ──
    results = []
    for i, img_path in enumerate(image_files, 1):
        sheet_num = img_path.stem.replace("test_sheet_", "")
        gt = ground_truth.get(sheet_num, {})

        print(f"[{i:02d}/{len(image_files)}] {img_path.name}", end="  ")
        result = evaluate_single_image(img_path, gt, all_answer_keys)
        results.append(result)

        # In dòng tóm tắt
        ans_acc = (result['answers_correct'] / result['answers_total'] * 100
                   if result['answers_total'] > 0 else 0)
        status = "✓" if result['success'] else "✗"
        pred = result['exam_code_predicted']
        gt_code = result['exam_code_gt']
        ma_de_str = "✓" if result['exam_code_correct'] else f"✗({pred}≠{gt_code})"
        sbd_str   = "✓" if result['student_id_correct']  else "✗"
        print(
            f"{status} | "
            f"anchor={'✓' if result['anchor_detected'] else '✗'}  "
            f"transform={'✓' if result['transform_success'] else '✗'}  "
            f"mã_đề={ma_de_str}  "
            f"SBD={sbd_str}  "
            f"đáp_án={result['answers_correct']}/{result['answers_total']}({ans_acc:.0f}%)  "
            f"time={result['processing_time']:.2f}s"
        )
        if result['errors']:
            for err in result['errors']:
                print(f"         ⚠ {err}")

    # ── Tính metrics ──
    print()
    print("=" * 80)
    metrics = calculate_metrics(results)

    # ── Đếm test cases & dòng code ──
    num_test_cases  = count_pytest_tests(base_dir)
    num_test_lines  = count_test_code_lines(base_dir)
    num_exam_codes  = len(all_answer_keys)

    # ══════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 78 + "╗")
    print("║{:^78}║".format("KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG"))
    print("╠" + "═" * 78 + "╣")

    def row(label, value, note=""):
        line = f"  {label:<42} {value:<14} {note}"
        print(f"║ {line:<76} ║")

    row("Tổng số ảnh kiểm thử",
        f"{metrics['total_images']} phiếu",
        "Đa dạng góc chụp, ánh sáng")

    row("Tỉ lệ detect anchor thành công",
        f"{metrics['anchor_detection_rate']:.1f}%"
        f" ({metrics['anchor_success_count']}/{metrics['total_images']})",
        "Fallback tọa độ cứng khi thất bại")

    row("Tỉ lệ Perspective Transform thành công",
        f"{metrics['transform_success_rate']:.1f}%"
        f" ({metrics['transform_success_count']}/{metrics['total_images']})",
        "Thất bại → giữ ảnh gốc")

    row("Độ chính xác nhận dạng bubble (đáp án)",
        f"{metrics['answer_accuracy']:.1f}%"
        f" ({metrics['correct_answers']}/{metrics['total_answers']})",
        f"Trên {metrics['total_images']} phiếu thực tế")

    row("Độ chính xác đọc mã đề",
        f"{metrics['exam_code_accuracy']:.1f}%"
        f" ({metrics['exam_code_correct_count']}/{metrics['total_images']})",
        "HoughCircles + Z-score, 3 chữ số")

    row("Độ chính xác đọc SBD",
        f"{metrics['student_id_accuracy']:.1f}%"
        f" ({metrics['student_id_correct_count']}/{metrics['total_images']})",
        "HoughCircles + Z-score")

    row("Thời gian xử lý trung bình / ảnh",
        f"~{metrics['avg_processing_time']:.1f}s",
        f"(min={metrics['min_processing_time']:.1f}s, max={metrics['max_processing_time']:.1f}s), CPU")

    row("Số mã đề hỗ trợ",
        f"{num_exam_codes} mã",
        "Đọc tự động từ phiếu")

    row("Tổng test cases tự động",
        f"{num_test_cases} tests",
        f"pytest, ~{num_test_lines} dòng test code")

    print("╚" + "═" * 78 + "╝")

    # ── In bảng Markdown (copy vào Word/LaTeX) ──
    print()
    print("=" * 80)
    print("  BẢNG MARKDOWN — Copy vào báo cáo đồ án")
    print("=" * 80)
    print()
    print("| Chỉ số đánh giá | Giá trị đạt được | Ghi chú |")
    print("|---|---|---|")
    print(f"| Độ chính xác nhận dạng bubble (đáp án) | **{metrics['answer_accuracy']:.1f}%** ({metrics['correct_answers']}/{metrics['total_answers']} ô) | Trên {metrics['total_images']} phiếu đa dạng điều kiện |")
    print(f"| Độ chính xác đọc mã đề | **{metrics['exam_code_accuracy']:.1f}%** ({metrics['exam_code_correct_count']}/{metrics['total_images']}) | Phụ thuộc chất lượng ảnh vùng mã đề |")
    print(f"| Độ chính xác đọc SBD | **{metrics['student_id_accuracy']:.1f}%** ({metrics['student_id_correct_count']}/{metrics['total_images']}) | HoughCircles + Z-score |")
    print(f"| Thời gian xử lý / ảnh | **~{metrics['avg_processing_time']:.1f} giây** | CPU, không dùng GPU |")
    print(f"| Tỉ lệ detect anchor thành công | **{metrics['anchor_detection_rate']:.1f}%** ({metrics['anchor_success_count']}/{metrics['total_images']}) | Fallback về tọa độ cứng khi thất bại |")
    print(f"| Tỉ lệ Perspective Transform thành công | **{metrics['transform_success_rate']:.1f}%** ({metrics['transform_success_count']}/{metrics['total_images']}) | Thất bại → giữ ảnh gốc |")
    print(f"| Số mã đề hỗ trợ | **{num_exam_codes} mã** | Đọc tự động từ phiếu |")
    print(f"| Tổng test cases tự động | **{num_test_cases} tests** | pytest, ~{num_test_lines} dòng test code |")
    print()

    # ── Lưu file kết quả ──
    json_out = output_dir / "evaluation_results.json"
    csv_out  = output_dir / "evaluation_results.csv"

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'details': results}, f, indent=2, ensure_ascii=False)

    export_csv(results, csv_out)

    print(f"💾 JSON chi tiết : {json_out}")
    print(f"💾 CSV chi tiết  : {csv_out}")
    print()


if __name__ == "__main__":
    main()
