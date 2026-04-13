"""
Hệ thống chấm trắc nghiệm tự động (OMR) - Optical Mark Recognition.

File chính để chạy pipeline xử lý: Đọc ảnh -> Tiền xử lý -> Nắn chỉnh -> Chấm điểm.

Author: Computer Vision Team
Date: 2026
"""

import sys
from pathlib import Path

# Import các module xử lý
from src.preprocessing import load_image, convert_to_grayscale, apply_noise_filter
from src.transform import detect_edges, find_document_corners, apply_perspective_transform
from src.grader import extract_bubble_grid, segment_bubbles, calculate_score


def main(image_path: str, answer_key_path: str = None) -> None:
    """
    Hàm chính để chạy pipeline chấm trắc nghiệm tự động.
    
    Pipeline gồm các bước:
    1. Đọc ảnh từ file
    2. Tiền xử lý: Chuyển sang xám và khử nhiễu
    3. Phát hiện biên và tìm 4 góc tờ giấy thi
    4. Nắn chỉnh ảnh về mặt phẳng chuẩn
    5. Trích xuất vùng đáp án và phân đoạn
    6. Chấm điểm và in kết quả
    
    Args:
        image_path (str): Đường dẫn đến file ảnh bài thi cần chấm.
        answer_key_path (str, optional): Đường dẫn đến file đáp án chuẩn (JSON hoặc TXT).
                                        Nếu None, sử dụng đáp án mẫu.
    
    Returns:
        None: Hàm in kết quả ra terminal.
    
    Examples:
        >>> python main.py data/test_sheet_01.jpg data/answer_key.json
        ========================================
        KẾT QUẢ CHẤM TRẮC NGHIỆM TỰ ĐỘNG
        ========================================
        Số câu đúng: 35/40
        Điểm số: 8.75/10
        ========================================
    """
    print("=" * 50)
    print("HỆ THỐNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)")
    print("=" * 50)
    
    # TODO: Bước 1 - ĐỌC ẢNH
    # TODO: 1.1 - Gọi load_image(image_path) để đọc ảnh
    # TODO: 1.2 - In thông báo: "✓ Đã đọc ảnh thành công"
    # TODO: 1.3 - Xử lý exception nếu file không tồn tại
    
    # TODO: Bước 2 - TIỀN XỬ LÝ
    # TODO: 2.1 - Gọi convert_to_grayscale(image) để chuyển sang ảnh xám
    # TODO: 2.2 - In thông báo: "✓ Đã chuyển sang ảnh xám"
    # TODO: 2.3 - Gọi apply_noise_filter(gray_image, filter_type="gaussian", kernel_size=5)
    # TODO: 2.4 - In thông báo: "✓ Đã áp dụng bộ lọc khử nhiễu"
    
    # TODO: Bước 3 - PHÁT HIỆN BIÊN VÀ TÌM GÓC
    # TODO: 3.1 - Gọi detect_edges(blurred_image) để phát hiện biên
    # TODO: 3.2 - In thông báo: "✓ Đã phát hiện biên"
    # TODO: 3.3 - Gọi find_document_corners(edges) để tìm 4 góc tờ giấy
    # TODO: 3.4 - In thông báo: "✓ Đã tìm thấy 4 góc tờ giấy thi"
    # TODO: 3.5 - Xử lý exception nếu không tìm thấy tờ giấy
    
    # TODO: Bước 4 - NẮN CHỈNH ẢNH
    # TODO: 4.1 - Gọi apply_perspective_transform(image, corners) để nắn chỉnh
    # TODO: 4.2 - In thông báo: "✓ Đã nắn chỉnh ảnh"
    
    # TODO: Bước 5 - TRÍCH XUẤT VÀ PHÂN ĐOẠN
    # TODO: 5.1 - Chuyển warped_image sang xám nếu cần
    # TODO: 5.2 - Gọi extract_bubble_grid(warped_gray) để cắt vùng ROI
    # TODO: 5.3 - In thông báo: "✓ Đã trích xuất vùng đáp án"
    # TODO: 5.4 - Gọi segment_bubbles(grid_image) để phân ngưỡng
    # TODO: 5.5 - In thông báo: "✓ Đã phân đoạn các ô trắc nghiệm"
    
    # TODO: Bước 6 - CHẤM ĐIỂM
    # TODO: 6.1 - Đọc đáp án chuẩn từ answer_key_path (nếu có)
    #             Hoặc sử dụng đáp án mẫu:
    #             answer_key = {1: 'A', 2: 'C', 3: 'B', 4: 'D', ...}
    # TODO: 6.2 - Gọi calculate_score(segmented_image, answer_key) để chấm điểm
    # TODO: 6.3 - Unpack kết quả: correct_count, score, student_answers = calculate_score(...)
    
    # TODO: Bước 7 - IN KẾT QUẢ
    # TODO: 7.1 - In header kết quả
    # TODO: 7.2 - In số câu đúng: f"Số câu đúng: {correct_count}/{len(answer_key)}"
    # TODO: 7.3 - In điểm số: f"Điểm số: {score:.2f}/10"
    # TODO: 7.4 - (Tùy chọn) In chi tiết từng câu:
    #             for question_num in sorted(student_answers.keys()):
    #                 student_ans = student_answers[question_num]
    #                 correct_ans = answer_key.get(question_num, '?')
    #                 status = "✓" if student_ans == correct_ans else "✗"
    #                 print(f"  Câu {question_num}: {student_ans} (Đáp án: {correct_ans}) {status}")
    # TODO: 7.5 - In footer
    
    # TODO: Bước 8 - (Tùy chọn) LƯU KẾT QUẢ
    # TODO: 8.1 - Lưu ảnh đã xử lý vào thư mục output/
    # TODO: 8.2 - Lưu kết quả chấm điểm vào file JSON hoặc CSV
    
    pass


if __name__ == "__main__":
    # TODO: Xử lý command-line arguments
    # TODO: Kiểm tra số lượng arguments: python main.py <image_path> [answer_key_path]
    # TODO: Nếu thiếu arguments, in hướng dẫn sử dụng:
    #       print("Cách sử dụng: python main.py <đường_dẫn_ảnh> [đường_dẫn_đáp_án]")
    #       print("Ví dụ: python main.py data/test_sheet_01.jpg data/answer_key.json")
    #       sys.exit(1)
    # TODO: Gọi hàm main() với các arguments
    
    # Ví dụ đơn giản (hardcode để test):
    # main("data/test_sheet_01.jpg")
    
    print("\n⚠️  Chưa triển khai xử lý command-line arguments.")
    print("Vui lòng uncomment dòng main() ở trên và cung cấp đường dẫn ảnh.")
