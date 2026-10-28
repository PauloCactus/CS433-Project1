# CS-433: Machine Learning - Project 1

This repository contains the implementation of standard machine learning methods and models for Project 1 of EPFL's CS-433 Machine Learning course.

## Project Structure

```text
├── implementations.py         # 6 required ML methods
├── helpers.py                 # Data loading and helper routines
├── run.py                     # Script generating final predictions
├── grading_tests/
│   ├── INSTRUCTIONS.md
│   ├── conftest.py
│   ├── test_project1_public.py
│   └── environment.yml
└── README.md
```

---

## Environment Setup

Set up the required environment using Conda:

```bash
conda create -n ml_proj1 python=3.11 numpy matplotlib pytest seaborn GitPython black
conda activate ml_proj1
```

---

## Reproducing Final Predictions (`run.py`)

Place the BRFSS dataset files (`x_train.csv`, `y_train.csv`, `x_test.csv`) inside the `./dataset/` directory and run:

```bash
python run.py
```

This script executes the complete pipeline and outputs the final test predictions to `submission.csv`:

1. **Feature Selection (`variable_columns`)**: Selects 47 clinically relevant cardiovascular, demographic, and lifestyle features (`cvd_variables`) out of the 321 raw survey columns based on the 2015 CDC BRFSS codebook.
2. **Survey Code Cleaning (`preprocess_cols`)**: Maps BRFSS special survey codes representing zero/none (`88`, `888`) to `0`, and missing/refused/unknown codes (`7`, `9`, `77`, `98`, `99`, `777`, `999`, `7777`, `9999`, `3`, `14`) to `NaN`.
3. **Imputation & Standardization (`preprocessing_X`, `preprocessing_Y`)**: Removes all-`NaN` and constant columns, imputes remaining `NaN` values using the training set column-wise median, standardizes features to zero mean and unit variance using training statistics, prepends an intercept column of ones, and maps labels $y \in \{-1, 1\}$ to $\{0, 1\}$.
4. **Cross-Validation & Hyperparameter Tuning (`tune_hyperparams`, `cv_reg`)**: Evaluates parameters using 5-fold cross-validation (`KFOLD = 5`, `SEED = 67`) with fold-local preprocessing (preventing data leakage) to maximize the mean F1 score. Optimal hyperparameters: $\lambda = 10^{-6}$, $\gamma = 0.775$, decision threshold $\tau = 0.20$, and `MAX_ITERS = 1000`.
5. **Training & Inference (`optimization`, `prediction`)**: Trains $\ell_2$-regularized logistic regression (`reg_logistic_regression`) on the full preprocessed training set and generates binary test predictions in $\{-1, 1\}$ saved to `submission.csv` (achieving a 5-fold CV F1 score of `0.4199 ± 0.0030` and accuracy of `87.15% ± 0.05%`).

---

## Running the Tests


### Official Grading Tests (`grading_tests`)

The public grading suite verifies function signatures, execution steps, docstrings, formatting, and file structure.

#### Local Tests
To test quickly without pushing commits to GitHub, pass `.` as the link and filter out the URL format check:

```bash
# From the project root:
PYTHONPATH=. pytest -v -k "not test_github_link" --github_link . grading_tests
```

#### Full Test From Official Course Repo
Run the tests after cloning the official course GitHub repository

```bash
# To clone the official course repository
git clone https://github.com/epfml/ML_course
```

```bash
# 1. Format code with black
black implementations.py run.py helpers.py

# 2. Push your changes
git add -u
git commit -m "Your commit message"
git push

# 3. After cloning the official course repository on your machine, navigate to grading_tests and run the following pytest command:
cd projects/project1/grading_tests 
pytest --github_link https://github.com/PauloCactus/CS433-Project1/tree/<commit_hash> .
```

> **Note on Commit URLs:**  
> The URL must point to a specific commit (`.../tree/<commit_hash>`). You can easily get the hash of a specific commit from the GitHub project home page by clicking on commits, and then clicking on the copy icon next to the commit you want to test.
