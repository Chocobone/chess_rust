import pandas as pd

def build_uci_vocabulary():
    """모든 가능한 UCI 이동과 특수 토큰을 포함한 사전을 생성합니다."""
    
    # 1. 특수 토큰 정의
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<1-0>", "<0-1>", "<1/2-1/2>"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # 체스판의 파일(열)과 랭크(행)
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f + r for f in files for r in ranks] # a1 ~ h8 (64개)
    
    current_idx = len(vocab)
    
    # 2. 일반적인 이동 (64 * 64 = 4096개)
    # 실제 체스에서 불가능한 기물 이동(예: a1h8 등)도 포함되지만, 
    # 어휘 사전 크기(~4200)가 작아 딥러닝 모델 임베딩에 부담이 없으므로 모두 생성합니다.
    for start in squares:
        for end in squares:
            if start != end:
                move = start + end
                vocab[move] = current_idx
                current_idx += 1
                
    # 엔진에서 패스를 나타낼 때 종종 쓰이는 널 이동(Null move)
    vocab["0000"] = current_idx
    current_idx += 1

    # 3. 프로모션 이동 (폰이 끝까지 갔을 때 퀸, 룩, 비숍, 나이트로 승급)
    promotions = ['q', 'r', 'b', 'n']
    
    # 백의 프로모션 (7랭크 -> 8랭크)
    for f1 in files:
        for f2 in files:
            # 폰은 직진하거나 대각선으로 잡으면서 전진(파일 차이가 1 이하)
            if abs(ord(f1) - ord(f2)) <= 1:
                start_sq = f1 + '7'
                end_sq = f2 + '8'
                for p in promotions:
                    vocab[start_sq + end_sq + p] = current_idx
                    current_idx += 1
                    
    # 흑의 프로모션 (2랭크 -> 1랭크)
    for f1 in files:
        for f2 in files:
            if abs(ord(f1) - ord(f2)) <= 1:
                start_sq = f1 + '2'
                end_sq = f2 + '1'
                for p in promotions:
                    vocab[start_sq + end_sq + p] = current_idx
                    current_idx += 1

    # 디코딩(ID -> 문자열)을 위한 역방향 사전도 함께 반환
    inverse_vocab = {idx: token for token, idx in vocab.items()}
    
    return vocab, inverse_vocab

def tokenize_game(uci_moves_str, result, vocab):
    """
    UCI 형태의 문자열 기보를 정수 ID 리스트로 변환합니다.
    형식: <SOS> [move1] [move2] ... [Result] <EOS>
    """
    tokens = [vocab["<SOS>"]]
    
    # 띄어쓰기를 기준으로 이동 분리 및 토큰화
    for move in str(uci_moves_str).split():
        tokens.append(vocab.get(move, vocab["<UNK>"]))
        
    # 결과 토큰 추가
    if result in ["1-0", "0-1", "1/2-1/2"]:
        tokens.append(vocab[result])
    else:
        tokens.append(vocab["<UNK>"])
        
    tokens.append(vocab["<EOS>"])
    
    return tokens

# ==========================================
# 실행 및 적용 예시
# ==========================================

# 1. 어휘 사전 생성
vocab, inverse_vocab = build_uci_vocabulary()
print(f"생성된 어휘 사전 크기: {len(vocab)} 토큰")
# 출력 예: 생성된 어휘 사전 크기: 4278 토큰

# 2. 이전에 추출한 CSV 데이터 불러오기 (가정)
# df = pd.read_csv("all_lichess_uci_data.csv")

# 임시 데이터프레임 예시
data = {
    "Result": ["1-0", "0-1"],
    "UCI_Moves": [
        "e2e4 c7c5 g1f3", # 짧은 기보 예시
        "d2d4 d7d5 c2c4 c7c6"
    ]
}
df = pd.DataFrame(data)

# 3. 데이터프레임의 데이터를 토큰화 적용
df["Tokenized"] = df.apply(lambda row: tokenize_game(row["UCI_Moves"], row["Result"], vocab), axis=1)

print("\n[토큰화 결과 확인]")
for i, row in df.iterrows():
    print(f"원본 기보: {row['UCI_Moves']} | 결과: {row['Result']}")
    print(f"토큰 ID: {row['Tokenized']}\n")

# 4. 검증 (디코딩)
# 토큰 ID가 다시 원래 문자열로 잘 복원되는지 확인
sample_tokens = df["Tokenized"].iloc[0]
decoded = [inverse_vocab[t_id] for t_id in sample_tokens]
print(f"디코딩 검증: {' '.join(decoded)}")
