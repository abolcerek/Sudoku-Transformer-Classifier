import pandas as pd
import numpy as np

# Load the original Sudoku dataset, dataset is located at https://www.kaggle.com/datasets/rohanrao/sudoku
df = pd.read_csv("sudoku.csv")

# Convert '.' which are represented as empty cells to '0'
df["puzzle"] = df["puzzle"].str.replace(".", "0")

# Check if a number is valid at the given cell in the board
def is_valid(board, row, col, num):
    # Checking the row and column
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    # Checking the 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

# Solving the puzzle through backtracking and count how many recursive steps it takes
def solve_sudoku(board):
    global attempt_counter
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        attempt_counter += 1  # Count each successful placement attempt
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0  # Undo the move if it leads to a dead end
                return False  # Backtracking
    return True  # Puzzle was solved

# Converting the 81-character puzzle string into a 9x9 array
def puzzle_to_grid(puzzle_str):
    return np.array([int(c) for c in puzzle_str], dtype=np.int64).reshape(9, 9)


# List for holding the attempt count for each puzzle
attempts_list = []

# Loop through every puzzle in the dataset
for idx, row in df.iterrows():
    puzzle = row["puzzle"]
    grid = puzzle_to_grid(puzzle)
    attempt_counter = 0  # Reset the global counter for each puzzle
    solve_sudoku(grid.copy())  # Use a copy so that we dont alter the original grid
    attempts_list.append(attempt_counter)  # Store the total attempts for this puzzle

df["attempts"] = attempts_list

# Save the updated DataFrame to a new CSV 
df.to_csv("SUDOKU.csv", index=False)


