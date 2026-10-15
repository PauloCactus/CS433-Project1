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
cd grading_tests
pytest --github_link https://github.com/PauloCactus/CS433-Project1/tree/<commit_hash> .
```

> **Note on Commit URLs:**  
> The URL must point to a specific commit (`.../tree/<commit_hash>`). You can easily get the hash of a specific commit from the GitHub project home page by clicking on commits, and then clicking on the copy icon next to the commit you want to test.
