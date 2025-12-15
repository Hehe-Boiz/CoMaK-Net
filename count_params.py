import torch
from CoMakNet import CoMakNet_tiny


def count_parameters(model_path, num_classes=5):
    # 1. Khởi tạo kiến trúc rỗng (Skeleton)
    # Lưu ý: num_classes phải khớp với lúc bạn train (RetinaMNIST là 5)
    print("-> Đang khởi tạo mô hình...")
    model = CoMakNet_tiny(num_classes=num_classes)

    # 2. Load file .pth vào bộ nhớ
    print(f"-> Đang đọc file: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')  # Load vào CPU cho nhẹ

    # Trích xuất state_dict (vì code save của bạn lưu cả optimizer, epoch...)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load trọng số vào mô hình
    try:
        model.load_state_dict(state_dict)
        print("-> Load trọng số thành công!")
    except RuntimeError as e:
        print(f"Lỗi kích thước (thường do sai num_classes): {e}")
        return

    # 3. [WHITE-BOX] Thuật toán đếm
    # model.parameters() trả về một iterator qua tất cả các tensor W, b
    # p.numel() trả về số lượng phần tử (number of elements) trong tensor đó
    # p.requires_grad kiểm tra xem tham số đó có bị đóng băng (frozen) không

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 30)
    print(f"📊 KẾT QUẢ ĐẾM TRỌNG SỐ:")
    print(f"   • Tổng số tham số (Total Params):     {total_params:,}")
    print(f"   • Tham số học được (Trainable Params): {trainable_params:,}")
    print(f"   • Kích thước ước tính (MB):            {total_params * 4 / 1024 / 1024:.2f} MB")
    # (Nhân 4 vì mỗi float32 tốn 4 bytes)
    print("=" * 30)


if __name__ == "__main__":
    # Thay đường dẫn tới file .pth thực tế của bạn
    ckpt_path = "./checkpoints/CoMakNet_tiny_retinamnist_auc_79.pth"
    count_parameters(ckpt_path, num_classes=5)