import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 앞서 작성된 토큰화 스크립트에서 어휘 사전을 가져옵니다.
from preprocessing.tokenize import build_uci_vocabulary


# 1. 보드 상태 추출 함수 (요구사항 반영)
def get_board_features(board):
    """
    현재 보드 상태와 부가 정보(캐슬링, 50수, 3수 동형)를 추출합니다.
    """
    # 1-1. 64개 칸의 기물 상태 (0: 빈칸, 1~6: 백, 7~12: 흑)
    squares = np.zeros(64, dtype=np.int64)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            # piece.piece_type은 1(Pawn)~6(King). 흑이면 +6
            squares[i] = (
                piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
            )

    # 1-2. 부가 정보 (Global State)
    # [백 킹사이드 캐슬링, 백 퀸사이드, 흑 킹사이드, 흑 퀸사이드, 50수 규칙(반수), 3수 동형 여부]
    global_state = np.zeros(6, dtype=np.float32)
    global_state[0] = int(board.has_kingside_castling_rights(chess.WHITE))
    global_state[1] = int(board.has_queenside_castling_rights(chess.WHITE))
    global_state[2] = int(board.has_kingside_castling_rights(chess.BLACK))
    global_state[3] = int(board.has_queenside_castling_rights(chess.BLACK))
    global_state[4] = board.halfmove_clock / 100.0  # 100반수(50수) 기준 정규화
    global_state[5] = int(board.can_claim_threefold_repetition())

    return squares, global_state


# 2. PyTorch Dataset 구성
class ChessDataset(Dataset):
    def __init__(self, csv_path, vocab):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.data = []

        print("데이터셋을 구축합니다. (기보를 재생하며 보드 상태를 추출합니다...)")
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.df.iterrows():
            uci_moves = str(row["UCI_Moves"]).split()
            result_str = row["Result"]

            # 게임 결과 수치화 (백 승: 1, 흑 승: -1, 무승부: 0)
            if result_str == "1-0":
                game_value = 1.0
            elif result_str == "0-1":
                game_value = -1.0
            else:
                game_value = 0.0

            board = chess.Board()
            for move_uci in uci_moves:
                # 1. 현재 보드 상태 추출
                squares, global_state = get_board_features(board)

                # 2. 타겟 Policy (현재 상황에서 실제로 둔 수)
                target_policy = self.vocab.get(move_uci, self.vocab["<UNK>"])

                # 3. 타겟 Value (현재 차례인 플레이어 기준의 승률로 변환)
                # 턴이 백이면 game_value 그대로, 흑이면 부호 반전
                target_value = game_value if board.turn == chess.WHITE else -game_value

                # 데이터 추가
                self.data.append((squares, global_state, target_policy, target_value))

                # 보드 업데이트 (다음 수 진행)
                board.push_uci(move_uci)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        squares, global_state, target_policy, target_value = self.data[idx]
        return (
            torch.tensor(squares, dtype=torch.long),
            torch.tensor(global_state, dtype=torch.float32),
            torch.tensor(target_policy, dtype=torch.long),
            torch.tensor([target_value], dtype=torch.float32),
        )


# 3. Transformer Encoder 기반 체스 모델
class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super(ChessTransformer, self).__init__()

        # 기물 임베딩 (13가지: 빈칸 1개 + 백 6개 + 흑 6개)
        self.piece_embedding = nn.Embedding(13, d_model)
        # 위치 임베딩 (64칸)
        self.position_embedding = nn.Embedding(64, d_model)

        # 부가 정보(캐슬링 등 6개 특징)를 임베딩 (1개의 토큰으로 변환)
        self.global_embedding = nn.Linear(6, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Heads (Policy & Value)
        # 64칸 + 1(부가정보) = 65개의 시퀀스를 Flatten 하여 출력층에 전달
        self.flatten_dim = d_model * 65

        # Policy Head: 어떤 수를 둘 것인가 (Vocab Size 만큼의 로짓 출력)
        self.policy_head = nn.Linear(self.flatten_dim, vocab_size)

        # Value Head: 현재 상태의 승률 (-1 ~ 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # -1(패) ~ 1(승) 범위 출력
        )

    def forward(self, squares, global_state):
        # squares: [batch, 64], global_state: [batch, 6]
        batch_size = squares.size(0)

        # 1. 64개 칸에 대한 임베딩 생성 (기물 정보 + 위치 정보)
        positions = (
            torch.arange(64, device=squares.device).unsqueeze(0).expand(batch_size, 64)
        )
        x_squares = self.piece_embedding(squares) + self.position_embedding(
            positions
        )  # [batch, 64, d_model]

        # 2. 부가 정보 임베딩을 시퀀스에 추가 (CLS 토큰과 유사한 역할)
        x_global = self.global_embedding(global_state).unsqueeze(
            1
        )  # [batch, 1, d_model]

        # 3. 시퀀스 병합: [batch, 65, d_model]
        x = torch.cat([x_global, x_squares], dim=1)

        # 4. 트랜스포머 인코더 통과
        encoded = self.transformer(x)

        # 5. Flatten 후 두 개의 Head(Policy, Value)로 분기
        encoded_flat = encoded.reshape(batch_size, -1)

        policy_logits = self.policy_head(encoded_flat)
        value = self.value_head(encoded_flat)

        return policy_logits, value


# 4. 학습 루프 (Training Loop)
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")

    # 어휘 사전 생성 (기존 파일 이용)
    vocab, _ = build_uci_vocabulary()
    vocab_size = len(vocab)

    # 데이터셋 및 데이터로더 설정
    csv_path = "/local_datasets/yho7374/lichess/csv/all_data.csv"  # 데이터 필터링에서 생성한 파일명
    dataset = ChessDataset(csv_path, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 모델 초기화
    model = ChessTransformer(
        vocab_size=vocab_size, d_model=128, nhead=4, num_layers=4
    ).to(device)

    # 손실 함수 및 옵티마이저 (Policy는 CrossEntropy, Value는 MSE 사용)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_policy_loss = 0
        total_value_loss = 0

        for batch_idx, (
            squares,
            global_state,
            target_policy,
            target_value,
        ) in enumerate(dataloader):
            squares, global_state = squares.to(device), global_state.to(device)
            target_policy, target_value = (
                target_policy.to(device),
                target_value.to(device),
            )

            optimizer.zero_grad()

            # 모델 예측
            policy_logits, value_pred = model(squares, global_state)

            # Loss 계산
            loss_policy = criterion_policy(policy_logits, target_policy)
            loss_value = criterion_value(value_pred, target_value)

            # 전체 Loss (두 loss를 더해서 최적화)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()

            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

        print(
            f"Epoch {epoch + 1}/{epochs} | Policy Loss: {total_policy_loss / len(dataloader):.4f} | Value Loss: {total_value_loss / len(dataloader):.4f}"
        )

    print("학습 완료! 모델을 저장합니다.")
    torch.save(model.state_dict(), "transformer_chess_engine.pth")


if __name__ == "__main__":
    train_model()
