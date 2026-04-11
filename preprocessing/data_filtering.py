import glob
import os

import pandas as pd
from chess import pgn


def preprocess_folder_to_uci(folder_path, output_filename):
    # 1. 지정된 폴더 내의 모든 .pgn 파일 경로를 리스트로 가져오기
    # 예: folder_path가 'data'라면 'data/*.pgn'을 검색
    search_pattern = os.path.join(folder_path, "*.pgn")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(f"'{folder_path}' 폴더 내에 PGN 파일이 존재하지 않습니다.")
        return None

    print(f"총 {len(file_paths)}개의 PGN 파일을 찾았습니다. 전처리를 시작합니다...\n")

    all_games = []

    # 2. 찾은 모든 파일에 대해 반복 작업 수행
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"-> 처리 중: {file_name}")

        with open(file_path, "r", encoding="utf-8") as pgn_file:
            game_count = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # 파일의 끝에 도달하면 종료

                # 메타데이터 및 기보(UCI) 추출
                result = game.headers.get("Result", "N/A")
                eco = game.headers.get("ECO", "N/A")
                uci_moves = " ".join(move.uci() for move in game.mainline_moves())

                all_games.append({"Result": result, "ECO": eco, "UCI_Moves": uci_moves})
                game_count += 1

        print(f"   완료: {file_name} (추출된 게임 수: {game_count}개)")

    # 3. 추출된 전체 데이터를 DataFrame으로 변환 및 CSV 저장
    df_uci = pd.DataFrame(all_games)
    df_uci.to_csv(output_filename, index=False)

    print("\n모든 파일의 전처리가 완료되었습니다!")
    print(
        f"총 {len(df_uci)}개의 게임 데이터가 '{output_filename}' 파일로 저장되었습니다."
    )

    return df_uci


# --------------------------
# 실행 부분
# --------------------------
# PGN 파일들이 모여있는 폴더의 경로를 입력하세요.
# (현재 스크립트와 같은 경로에 'pgn_data'라는 폴더가 있다고 가정)
TARGET_FOLDER = "/local_datasets/yho7374/lichess/pgn"
OUTPUT_CSV_PATH = "/local_datasets/yho7374/lichess/csv/all_data.csv"

# 함수 실행
df = preprocess_folder_to_uci(TARGET_FOLDER, OUTPUT_CSV_PATH)

# 결과 일부 확인
if df is not None:
    print(df.head())
