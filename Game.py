import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from Transformer import SudokuTransformerClassifier  # loads model class from local file
import random

# Load the trained model
def load_model(model_path):
    model = SudokuTransformerClassifier()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
    model.eval()
    return model

# Tracks the user's recent performance to assign them a difficulty
class UserSkillTracker:
    def __init__(self):
        self.solved_puzzles = []

    def update_skill(self, difficulty_bin, time_taken):
        self.solved_puzzles.append((difficulty_bin, time_taken))
        if len(self.solved_puzzles) > 10:
            self.solved_puzzles.pop(0)

    def estimate_skill_level(self):
        if not self.solved_puzzles:
            return 1  # default to medium
        difficulties = [entry[0] for entry in self.solved_puzzles]
        return round(np.mean(difficulties))

# Validating the Sudoku board
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

# Backtracking Sudoku solver
def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

# Format board into a single string for prediction
def board_to_string(board):
    return ''.join(str(cell) if cell != 0 else '.' for row in board for cell in row)

# Convert string to a 9x9  array
def convert_to_array(puzzle_str):
    return np.array([int(c) if c.isdigit() else 0 for c in puzzle_str]).reshape(9, 9)

# Estimate difficulty using the model 
def classify_difficulty(model, puzzle_str):
    array = convert_to_array(puzzle_str)
    array = torch.tensor(array, dtype=torch.long).view(1, 9, 9)
    with torch.no_grad():
        output = model(array)
        return torch.argmax(output, dim=1).item()

# Random puzzle generator
def generate_random_puzzle_with_difficulty(model, target_bin=None):
    attempts = 0
    while True:
        board = [[0 for _ in range(9)] for _ in range(9)]
        filled = 0
        while filled < random.randint(20, 40):
            r = random.randint(0, 8)
            c = random.randint(0, 8)
            if board[r][c] == 0:
                candidates = list(range(1, 10))
                random.shuffle(candidates)
                for num in candidates:
                    if is_valid(board, r, c, num):
                        board[r][c] = num
                        filled += 1
                        break
        puzzle_str = board_to_string(board)
        grid_copy = convert_to_array(puzzle_str)
        if not solve_sudoku(grid_copy.copy()):
            continue
        difficulty_bin = classify_difficulty(model, puzzle_str)
        if target_bin is None or difficulty_bin == target_bin:
            return puzzle_str, difficulty_bin
        attempts += 1
        if attempts > 100:
            return puzzle_str, difficulty_bin

# GUI app class
def build_gui():
    class SudokuApp:
        def __init__(self, root, model, user_tracker):
            self.root = root
            self.model = model
            self.user_tracker = user_tracker
            self.start_time = time.time()
            self.selected_entry = None
            self.root.title("Sudoku")

            self.timer_label = tk.Label(self.root, text="Time: 0s", font=("Helvetica", 16))
            self.timer_label.grid(row=0, column=0, columnspan=9, pady=(10, 0))

            self.generate_new_puzzle()

            btn_frame = tk.Frame(self.root)
            btn_frame.grid(row=12, column=0, columnspan=9, pady=20)

            tk.Button(btn_frame, text="Submit", font=("Arial", 16), width=12, command=self.submit_solution).pack(side="left", padx=10)
            tk.Button(btn_frame, text="Show Solution", font=("Arial", 16), width=12, command=self.show_solution).pack(side="left", padx=10)
            tk.Button(btn_frame, text="New Puzzle", font=("Arial", 16), width=12, command=self.generate_new_puzzle).pack(side="left", padx=10)
            tk.Button(btn_frame, text="Exit", font=("Arial", 16), width=12, command=self.root.destroy).pack(side="left", padx=10)

            self.update_timer()

        def generate_new_puzzle(self):
            estimated_skill = self.user_tracker.estimate_skill_level()
            self.puzzle, self.difficulty_bin = generate_random_puzzle_with_difficulty(self.model, target_bin=estimated_skill)
            self.board = convert_to_array(self.puzzle)
            self.solved_board = self.board.copy()
            solve_sudoku(self.solved_board)
            self.create_grid()
            self.start_time = time.time()

        def create_grid(self):
            for widget in self.root.winfo_children():
                if not isinstance(widget, tk.Frame) and not isinstance(widget, tk.Label):
                    widget.destroy()

            self.canvas = tk.Canvas(self.root, width=540, height=540, bg="white", highlightthickness=0)
            self.canvas.grid(row=1, column=0, columnspan=9)
            cell_size = 60

            for row_block in range(3):
                for col_block in range(3):
                    x0 = col_block * 3 * cell_size
                    y0 = row_block * 3 * cell_size
                    self.canvas.create_rectangle(
                        x0, y0, x0 + 3 * cell_size, y0 + 3 * cell_size,
                        fill="#f6f6f6" if (row_block + col_block) % 2 == 0 else "white",
                        width=0
                    )
            for i in range(10):
                width = 3 if i % 3 == 0 else 1
                self.canvas.create_line(0, i * cell_size, 540, i * cell_size, width=width, fill="black")
                self.canvas.create_line(i * cell_size, 0, i * cell_size, 540, width=width, fill="black")

            self.entries = [[None for _ in range(9)] for _ in range(9)]
            for row in range(9):
                for col in range(9):
                    value = self.board[row][col]
                    entry = tk.Entry(
                        self.canvas,
                        width=2,
                        font=("Helvetica", 20),
                        justify="center",
                        bd=0,
                        highlightthickness=0,
                        relief="flat",
                        bg="white"
                    )
                    x0 = col * cell_size
                    y0 = row * cell_size
                    self.canvas.create_window(x0 + cell_size/2, y0 + cell_size/2, window=entry, width=cell_size-14, height=cell_size-14)

                    if value != 0:
                        entry.insert(0, str(value))
                        entry.config(state="disabled", disabledbackground="#e0e0e0", fg="black")
                    else:
                        entry.config(bg="white", fg="black")
                    entry.bind("<Button-1>", lambda e, r=row, c=col: self.select_cell(r, c))
                    self.entries[row][col] = entry

        def select_cell(self, row, col):
            if self.selected_entry:
                prev_row, prev_col = self.selected_entry
                prev_entry = self.entries[prev_row][prev_col]
                if self.board[prev_row][prev_col] != 0:
                    prev_entry.config(disabledbackground="#e0e0e0")
                else:
                    prev_entry.config(bg="white")
            entry = self.entries[row][col]
            if self.board[row][col] != 0:
                entry.config(disabledbackground="#cce5ff")
            else:
                entry.config(bg="#cce5ff")
            self.selected_entry = (row, col)

        def submit_solution(self):
            user_solution = np.zeros((9, 9), dtype=int)
            for row in range(9):
                for col in range(9):
                    value = self.entries[row][col].get()
                    if not value.isdigit() or not (1 <= int(value) <= 9):
                        messagebox.showerror("Sudoku", "All cells must be filled with digits 1–9.")
                        return
                    user_solution[row][col] = int(value)
            if np.array_equal(user_solution, self.solved_board):
                time_taken = time.time() - self.start_time
                messagebox.showinfo("Sudoku", "Correct! You solved it in %.2f seconds." % time_taken)
                self.user_tracker.update_skill(self.difficulty_bin, time_taken)
                self.generate_new_puzzle()
            else:
                messagebox.showerror("Sudoku", "Incorrect solution. Please try again!")

        def show_solution(self):
            for row in range(9):
                for col in range(9):
                    entry = self.entries[row][col]
                    entry.config(state="normal")
                    entry.delete(0, tk.END)
                    entry.insert(0, str(self.solved_board[row][col]))
                    entry.config(state="disabled", disabledbackground="#d0ffd0", fg="black")

        def update_timer(self):
            elapsed = int(time.time() - self.start_time)
            self.timer_label.config(text="Time: " + str(elapsed) + "s")
            self.root.after(1000, self.update_timer)

    return SudokuApp

if __name__ == "__main__":
    model = load_model("Transformer.pth")
    user_tracker = UserSkillTracker()
    root = tk.Tk()
    SudokuApp = build_gui()
    app = SudokuApp(root, model, user_tracker)
    root.mainloop()
