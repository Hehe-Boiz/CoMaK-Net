import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from CoMakNet import CoMakNet_tiny, CoMakNet_large
from dataset import build_dataset
import timm

# Định nghĩa lại map model giống main.py
model_classes = {
    'CoMakNet_tiny': CoMakNet_tiny,
    'CoMakNet_large': CoMakNet_large
}


def test_retinalmnist(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Chuẩn bị Dữ liệu
    # build_dataset trả về (train_ds, test_ds, n_classes)
    # Ta chỉ quan tâm test_dataset
    print("Loading dataset...")
    _, test_dataset, n_classes = build_dataset(args)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 2. Khởi tạo Kiến trúc Mô hình (Model Skeleton)
    print(f"Building model: {args.model_name}")
    if args.model_name in model_classes:
        model = model_classes[args.model_name](num_classes=n_classes)
    else:
        # Fallback sang timm nếu không phải CoMaK-Net
        model = timm.create_model(args.model_name, num_classes=n_classes)

    model = model.to(device)

    # 3. Load Trọng số (Load Weights) từ Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.checkpoint_path)

    # [White-box Explan]: File .pth thường lưu cả optimizer, epoch...
    # Ta chỉ cần lấy phần 'model' (state_dict) chứa trọng số.
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint  # Trường hợp file chỉ lưu state_dict thuần

    # Load vào model (strict=True để đảm bảo kiến trúc khớp 100%)
    try:
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 4. Quá trình Test (Inference)
    model.eval()  # QUAN TRỌNG: Chuyển sang chế độ đánh giá
    y_score = torch.tensor([]).to(device)

    print("Start Testing...")
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm VRAM
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Với bài toán Multi-class classification của RetinalMNIST
            # Output cần được Softmax để ra xác suất (Probability)
            outputs = outputs.softmax(dim=-1)

            y_score = torch.cat((y_score, outputs), 0)

    # 5. Đánh giá bằng thư viện chuẩn MedMNIST
    # Chuyển về CPU để tính toán metric (tránh lỗi sklearn trên GPU)
    y_score = y_score.cpu().detach().numpy()

    evaluator = Evaluator('retinamnist', 'test', root='./data', size=224)
    metrics = evaluator.evaluate(y_score)

    print("=" * 30)
    print(f"📊 Test Result for {args.model_name}:")
    print(f"   AUC  : {metrics[0]:.4f}")
    print(f"   ACC  : {metrics[1]:.4f}")
    print("=" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CoMakNet_tiny')
    parser.add_argument('--dataset', type=str, default='retinamnist')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    test_retinalmnist(args)