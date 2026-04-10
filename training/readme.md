구현 상세 및 설명
보드 상태의 토큰화 (Transformer 입력용)

get_board_features()에서 64개의 각 체스 칸(Square)을 하나의 시퀀스로 처리하도록 변환합니다. 빈칸은 0, 백/흑의 기물은 1~12의 ID를 부여받습니다.

Transformer는 순서 정보를 스스로 알 수 없으므로 position_embedding을 더해 기물이 보드 어디에 위치해 있는지 모델이 알 수 있게 합니다.

부가 정보 (Global Context)

캐슬링, 50수 규칙(반수 계산), 3수 동형 정보는 별도의 global_state 벡터로 추출됩니다.

이 벡터는 Linear Layer를 거쳐 Transformer의 차원 크기로 변환된 후, 64개의 칸 정보(기물) 앞에 추가적인 1개의 토큰(CLS 토큰 역할)으로 붙여져 총 65길이의 시퀀스로 병합됩니다.

MCTS를 위한 Multi-Head 출력

트랜스포머 인코더를 통과한 결과를 펼쳐서(Flatten), 두 개의 출력을 내보냅니다.

Policy Head: 가능한 모든 UCI 이동(vocab 크기, 약 4200여개)에 대한 선택 확률 분포(Logits)를 반환합니다. MCTS는 트리 확장을 할 때 이 확률을 가중치(Prior)로 사용합니다.

Value Head: 현재 차례(Turn)의 플레이어 입장에서 이길 확률을 -1.0 ~ 1.0 (Tanh 사용) 사이의 값으로 평가합니다.

가능한 수 정보 (Legal Moves)

신경망 자체는 사전(vocab)에 있는 모든 수를 대상으로 확률을 출력하지만, MCTS 탐색 단계에서 모델의 출력을 사용할 때 python-chess의 board.legal_moves를 활용하여 불가능한 수의 확률은 마스킹(0으로 처리)하여 사용하게 됩니다.

이 코드를 실행하시면 CSV 파일의 기보를 재생하면서 각 수의 보드 상태와 게임의 결과를 매핑한 데이터셋이 구축되고 Transformer 모델 학습이 진행됩니다.
