a7import os
import pickle

from chess import pgn
from chess import Board

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def load_pgn(file_path):
    games = []
    with open (file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def board_to_matrix(board: Board):
    matrix = np.zeros ((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod (square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix [row, col, piece_type + piece_color] = 1
    return matrix

def create_input_for_nn (games):
    X=[]
    y=[]
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return X, y

def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int

#load games
files = [file for file in os.listdir("data") if file.endswith("pgn")]
games = []
i = 1
MAX_GAMES_TO_LOAD = 1 #45000
for file in files:
    games.extend(load_pgn(f"data/{file}"))
    print (f"files: {i}")
    print (f"games: {len(games)}")
    i += 1
    if len(games) > MAX_GAMES_TO_LOAD:
        break

X, y = create_input_for_nn(games)
y, move_to_int = encode_moves(y) #to do: save
y = to_categorical(y, num_classes=len(move_to_int))
X = np.array(X)

#save move_to_int
with open('move_to_int.pkl', 'wb') as outp:
    pickle.dump(move_to_int, outp, pickle.HIGHEST_PROTOCOL)


model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(move_to_int), activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=50, validation_split=0.1, batch_size=64)
model.save("models/chess_model.keras")


#read move_to_int
with open('move_to_int.pkl', 'rb') as inp:
    move_to_int = pickle.load(inp)

model = load_model("models/chess_model.keras")
#to do: load move_to_int
int_to_move = dict(zip(move_to_int.values(), move_to_int.keys()))


def predict_next_move(board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(predictions)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    return None

# Create a chess board (start position)
board = Board()
# Display the board before prediction
print("Initial board:")
print(board)

while not board.is_checkmate():

    # Predict and make the move
    next_move = predict_next_move(board)
    board.push_uci(next_move)

    # Display the board after prediction
    print("\nComputer's move:", next_move)
    print("Board after computer's move:")
    print(board)

    next_move = input("Enter your move and press enter to continue:")

    board.push_uci(next_move)
    print("\nYour move:", next_move)
    print("Board after your move:")
    print(board)