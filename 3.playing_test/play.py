#!/usr/bin/env python3
"""Terminal chess vs trained ChessTransformer model.

Usage:
    python play.py path/to/model_final.pt
    python play.py path/to/model_final.pt --color black --sims 400
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from chess_tokenize import build_uci_vocabulary, get_legal_move_mask
from transformer import ChessTransformer, encode_board, run_mcts

# unicode pieces: (black_symbol, white_symbol)
_PIECE_SYMBOLS = {
    chess.PAWN:   ("♟", "♙"),
    chess.KNIGHT: ("♞", "♘"),
    chess.BISHOP: ("♝", "♗"),
    chess.ROOK:   ("♜", "♖"),
    chess.QUEEN:  ("♛", "♕"),
    chess.KING:   ("♚", "♔"),
}


def render_board(board: chess.Board, flip: bool = False) -> str:
    ranks = range(7, -1, -1) if not flip else range(8)
    files = range(8) if not flip else range(7, -1, -1)
    file_label = "  " + " ".join("abcdefgh"[f] for f in files)

    lines = [file_label]
    for r in ranks:
        row = f"{r + 1} "
        for f in files:
            piece = board.piece_at(chess.square(f, r))
            if piece is None:
                light = (f + r) % 2 == 1
                row += ("· " if light else "  ")
            else:
                sym_b, sym_w = _PIECE_SYMBOLS[piece.piece_type]
                row += (sym_w if piece.color == chess.WHITE else sym_b) + " "
        row += f"{r + 1}"
        lines.append(row)
    lines.append(file_label)
    return "\n".join(lines)


def top_moves(root, board: chess.Board, n: int = 4) -> list:
    candidates = []
    for uci, child in root.children.items():
        if child.N == 0:
            continue
        try:
            san = board.san(chess.Move.from_uci(uci))
        except Exception:
            san = uci
        candidates.append((uci, san, child.N, child.Q()))
    candidates.sort(key=lambda x: -x[2])
    return candidates[:n]


def load_model(ckpt_path: str, vocab_size: int, device) -> ChessTransformer:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ChessTransformer(vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def play(ckpt_path: str, user_color: chess.Color, mcts_sims: int = 200) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, _ = build_uci_vocabulary()

    print(f"모델 로드: {ckpt_path}  (device: {device})")
    model = load_model(ckpt_path, len(vocab), device)

    board = chess.Board()
    flip = user_color == chess.BLACK

    color_str = "백(White)" if user_color == chess.WHITE else "흑(Black)"
    print(f"\n{'='*44}")
    print(f" 체스 대국 시작  —  플레이어: {color_str}")
    print(f"{'='*44}")

    while not board.is_game_over():
        print()
        print(render_board(board, flip=flip))

        if board.turn == user_color:
            _user_turn(board, user_color)
        else:
            _model_turn(board, model, vocab, device, mcts_sims)

    # 게임 종료
    print()
    print(render_board(board, flip=flip))
    _print_result(board, user_color)


def _user_turn(board: chess.Board, user_color: chess.Color) -> None:
    color_str = "백" if user_color == chess.WHITE else "흑"
    legal_uci = {m.uci() for m in board.legal_moves}
    examples = ", ".join(list(legal_uci)[:5])

    print(f"\n당신의 차례 ({color_str})  —  UCI 또는 SAN 표기 입력")
    while True:
        raw = input("이동 입력 (예: e2e4 / Nf3, 또는 quit): ").strip()
        if raw.lower() == "quit":
            print("게임을 종료합니다.")
            sys.exit(0)

        # UCI 시도
        if raw in legal_uci:
            board.push_uci(raw)
            return

        # SAN 시도
        try:
            move = board.parse_san(raw)
            if move in board.legal_moves:
                board.push(move)
                return
        except Exception:
            pass

        print(f"  유효하지 않은 수입니다. 합법적인 수 예시: {examples}")


def _model_turn(board: chess.Board, model, vocab, device, sims: int) -> None:
    print(f"\n모델 생각 중... (MCTS {sims}회)")
    root = run_mcts(board, model, vocab, device, sims=sims)
    candidates = top_moves(root, board, n=4)

    if not candidates:
        return

    total_visits = sum(c[2] for c in candidates)
    print("\n  후보 수     UCI      방문    승률")
    print("  " + "-" * 38)
    for i, (uci, san, n, q) in enumerate(candidates, 1):
        pct = n / total_visits * 100 if total_visits else 0
        bar = "█" * max(1, round(pct / 5))
        sign = "+" if q >= 0 else ""
        print(f"  {i}. {san:<8s} ({uci})  {n:4d}  {sign}{q:+.3f}  {bar}")

    best_uci, best_san = candidates[0][0], candidates[0][1]
    print(f"\n  모델 선택: {best_san} ({best_uci})")
    board.push_uci(best_uci)


def _print_result(board: chess.Board, user_color: chess.Color) -> None:
    outcome = board.outcome()
    print(f"\n{'='*44}")
    if outcome is None:
        print(" 게임 종료")
    elif outcome.winner == user_color:
        print(" 축하합니다! 승리하셨습니다! 🎉")
    elif outcome.winner is None:
        print(" 무승부입니다.")
    else:
        print(" 모델이 이겼습니다.")
    print(f" 결과: {board.result()}")
    print(f"{'='*44}")


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 체스 AI와 터미널 대국")
    parser.add_argument("checkpoint", help="모델 체크포인트 경로 (.pt)")
    parser.add_argument(
        "--color", choices=["white", "black"],
        help="플레이어 색상 (생략 시 직접 선택)",
    )
    parser.add_argument(
        "--sims", type=int, default=200,
        help="MCTS 시뮬레이션 횟수 (기본: 200)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다 — {args.checkpoint}")
        sys.exit(1)

    if args.color is None:
        while True:
            c = input("색상을 선택하세요 [white / black]: ").strip().lower()
            if c in ("white", "w"):
                args.color = "white"
                break
            if c in ("black", "b"):
                args.color = "black"
                break
            print("  'white' 또는 'black'을 입력하세요.")

    user_color = chess.WHITE if args.color == "white" else chess.BLACK
    play(args.checkpoint, user_color, mcts_sims=args.sims)


if __name__ == "__main__":
    main()
