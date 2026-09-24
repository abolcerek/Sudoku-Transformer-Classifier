# Sudoku Transformer Classifier

A transformer encoder that classifies 9×9 Sudoku puzzles as **easy**, **medium**, or **hard**, plus a Tkinter Sudoku game that generates random puzzles, classifies them with the model, and serves the player puzzles matched to their recent performance.

Difficulty labels are not human-annotated. They are derived from how much work a backtracking solver needs: each puzzle is solved exhaustively, the number of digit placements is counted, and puzzles are binned by the log of that count. The transformer then learns to predict that bin directly from the 81-cell puzzle string, without solving anything.

## Repository layout

| File | Purpose |
| --- | --- |
| `Preprocess.py` | Labels the raw Kaggle dataset with solver effort (`attempts`) and writes `SUDOKU.csv`. |
| `Transformer.py` | Defines `SudokuTransformerClassifier`, bins difficulty, and trains the model. |
| `Game.py` | Tkinter Sudoku app: random puzzle generation, model-based difficulty targeting, skill tracking. |

## How it works

### 1. Labeling (`Preprocess.py`)

Reads `sudoku.csv` ([Kaggle: 9 Million Sudoku Puzzles and Solutions](https://www.kaggle.com/datasets/rohanrao/sudoku)), replaces `.` with `0`, and runs a recursive backtracking solver over every puzzle while counting each successful digit placement. That count becomes the `attempts` column in `SUDOKU.csv` — a proxy for how much search the puzzle demands.

### 2. Binning (`Transformer.py`)

- Puzzles with `attempts > 100000` are dropped as outliers.
- `log_attempts = log1p(attempts)` flattens the heavy right skew.
- Cut points at **6.04** and **6.98** split puzzles into class `0` (easy), `1` (medium), `2` (hard).
- An 80/20 train/validation split, stratified by difficulty.

### 3. Model

Each of the 81 cells is a token drawn from a vocabulary of 10 (digits 0–9, where 0 means empty).

- `nn.Embedding(10, 64)` — cell-value embeddings
- A learned `[CLS]` token prepended to the sequence, plus a learned positional embedding of length 82
- 4 `TransformerEncoderLayer`s, 8 attention heads, feed-forward dim 256, dropout 0.1
- The `[CLS]` output feeds a `64 → 64 → 3` MLP head

Training uses AdamW at `lr=1e-3`, cross-entropy with `label_smoothing=0.1`, gradient clipping at norm 1.0, and a `CosineAnnealingWarmRestarts` schedule (`T_0=5, T_mult=2`) stepped per batch. Runs for 20 epochs on CUDA when available, and plots train/validation accuracy at the end.

### 4. The game (`Game.py`)

`UserSkillTracker` keeps the difficulty bins of your last 10 solved puzzles and rounds their mean into a target bin (defaulting to medium).

Puzzles are generated the standard way: `fill_board` produces a random complete solution, then `carve_puzzle` removes clues one at a time, putting a clue back whenever its removal would leave more than one solution. Carving stops at `MIN_GIVENS` (30) clues, which keeps generated boards in the same density range as the training data. Each candidate is classified, and generation retries until the predicted bin matches your target, falling back to the first candidate after `max_attempts` (20). The GUI has a live timer, Submit, Show Solution, New Puzzle, and Exit.

## Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
tqdm
```

`tkinter` ships with most CPython installs; on Linux you may need `python3-tk`.

```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm
```

## Usage

Download `sudoku.csv` from Kaggle into the repo root, then:

```bash
python Preprocess.py    # sudoku.csv -> SUDOKU.csv (slow: solves every puzzle)
python Transformer.py   # trains the classifier
python Game.py          # loads Transformer.pth and launches the GUI
```

## Implementation notes

A few things that are easy to get wrong here, and how the code handles them:

- **The checkpoint is saved after training.** `torch.save` runs once `train_model` returns, so `Transformer.pth` holds trained weights.
- **`Transformer.py` is import-safe.** Dataset loading, the split, and the training run live under `if __name__ == "__main__":`, so `Game.py`'s `from Transformer import SudokuTransformerClassifier` pulls in only the model class. Reusable pieces are exposed as `load_dataframe`, `build_loaders`, and `train_model`.
- **Puzzles stay strings.** Both scripts pass an explicit `dtype` to `read_csv`. Without it, pandas 3.x parses an 81-digit puzzle as an integer, which drops leading zeros and breaks `SudokuDataset`. `str.replace(".", "0", regex=False)` likewise keeps the dot literal.
- **`classify_difficulty` feeds the model a flat `(1, 81)` sequence,** matching what `forward` expects; a `(9, 9)` grid raises a dimension error at the `[CLS]` concatenation.
- **Generated puzzles have exactly one solution,** so the board the grader compares against is the only correct answer.

## Notes on the difficulty proxy

Backtracking attempt count measures brute-force search cost, not human difficulty — a puzzle that resists naive left-to-right backtracking may be easy for a person using standard techniques, and vice versa. The three bins should be read as "search effort tiers," which is what the classifier actually learns to predict.

`MIN_GIVENS` in `Game.py` is the one knob tying generated puzzles to the training distribution. It is set to 30 as a reasonable default; if you want a tighter match, check the clue-count distribution in your `SUDOKU.csv` and set it accordingly.
