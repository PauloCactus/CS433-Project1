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

## Running the Tests


### Official Grading Tests (`grading_tests`)

The public grading suite verifies function signatures, execution steps, docstrings, formatting, and file structure.

#### Fast Iteration (Local Directory)
To iterate quickly without pushing commits to GitHub, pass `.` as the link and filter out the URL format check:

```bash
# From the project root:
PYTHONPATH=. pytest -v -k "not test_github_link" --github_link . grading_tests
```

#### Full Test Against Remote GitHub Repo
Simulates the actual course grading system by cloning your repository at a specific commit hash:

```bash
# 1. Format code with black (recommended by course guidelines)
black implementations.py run.py helpers.py

# 2. Push your changes
git add -u
git commit -m "Your commit message"
git push

# 3. After cloning the official course repository on your machine, run the grading suite from inside grading_tests :
cd grading_tests
pytest --github_link https://github.com/PauloCactus/CS433-Project1/tree/<COMMIT HASH> .
```

> **Note on Commit URLs:**  
> The URL must point to a specific commit (`.../tree/<commit_hash>`). You can easily get this link by opening your repo on GitHub, pressing the <kbd>y</kbd> key on your keyboard, and copying the URL from the browser bar.
