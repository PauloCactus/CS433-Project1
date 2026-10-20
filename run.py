import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import *

# ---------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------------------------
SEED = 67
DATA_PATH = './dataset'
SUBMISSION_PATH = "submission.csv"

# Model hyperparameters (tune via local cross-validation)
LAMBDA = 1e-4
GAMMA = 0.1
MAX_ITERS = 1000

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. 3. Evaluation & Optimization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. Main Execution Flow
# ---------------------------------------------------------------------------

def main():
    print('Running main...')

    np.random.seed(SEED)

    print("1. Loading dataset...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)




if __name__ == "__main__":
    main()