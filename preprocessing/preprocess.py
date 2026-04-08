import chess.pgn
import pandas as pd

def preprocess_pgn_to_uci(file_paths):
    all_games = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as pgn_file:
            while True:
                # 게임 단위로 읽기 (메모리 효율성)
                game = chess.pgn.read_game(pgn_file)
                
                if game is None:
                    break  # 더 이상 읽을 게임이 없으면 종료

                # 1. 메타데이터 추출 (승패, ECO)
                result = game.headers.get("Result", "N/A")
                eco = game.headers.get("ECO", "N/A")
                
                # 2. 기보를 UCI 포맷으로 변환
                # mainline_moves()의 각 chess.Move 객체에 대해 .uci()를 호출합니다.
                # 예: 'e2e4 c7c5 g1f3 ...' 형태로 공백을 기준으로 연결합니다.
                uci_moves = " ".join(move.uci() for move in game.mainline_moves())

                all_games.append({
                    "Result": result,
                    "ECO": eco,
                    "UCI_Moves": uci_moves
                })
    
    return pd.DataFrame(all_games)

# 처리할 파일 리스트 정의 (업로드하신 파일명 기준)
files = ["lichess_elite_2016-09.pgn", "lichess_elite_2017-06.pgn"]

# 실행 및 결과 확인
print("데이터 전처리를 시작합니다. 파일 크기에 따라 시간이 소요될 수 있습니다...")
df_uci = preprocess_pgn_to_uci(files)

# 상위 5개 데이터 출력
print(df_uci.head())

# 모델 학습을 위해 CSV 파일로 저장 (추천)
# CSV로 저장해두면 이후 학습 스크립트에서 PGN을 파싱할 필요 없이 빠르게 로드 가능합니다.
df_uci.to_csv("lichess_uci_dataset.csv", index=False)
print("전처리 및 CSV 저장이 완료되었습니다.")