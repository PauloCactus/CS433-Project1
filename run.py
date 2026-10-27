import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data, create_csv_submission
from implementations import *

# ---------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------------------------
SEED = 67
DATA_PATH = "./dataset"
SUBMISSION_PATH = "submission.csv"

# Model hyperparameters (tune via local cross-validation)
LAMBDA = 1e-6
GAMMA = 0.775
MAX_ITERS = 1000
KFOLD = 5
THRESHOLD = 0.20

# Obtained from https://www2.stat.duke.edu/~cr173/Sta102_Sp16/Proj/codebook.html
cvd_variables = {
    # General health and cardiovascular history with code for NaN values or None (https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
    "GENHLTH": [7, 9],
    "PHYSHLTH": [88, 77, 99],
    "MENTHLTH": [88, 77, 99],
    "POORHLTH": [88, 77, 99],
    "BPHIGH4": [7, 9],
    "BPMEDS": [7, 9],
    "BLOODCHO": [7, 9],
    "CHOLCHK": [7, 9],
    "TOLDHI2": [7, 9],
    "CVDSTRK3": [7, 9],
    "ADDEPEV2": [7, 9],
    "CHCKIDNY": [7, 9],
    # Diabetes risk
    "DIABETE3": [7, 9],
    "DIABAGE2": [98, 99],
    # Demographics and major risk factors
    "SEX": [],
    "EDUCA": [9],
    "VETERAN3": [7, 9],
    "EMPLOY1": [9],
    "INCOME2": [77, 99],
    "WEIGHT2": [7777, 9999],
    "HEIGHT3": [7777, 9999],
    "QLACTLM2": [7, 9],
    "USEEQUIP": [7, 9],
    "DIFFWALK": [7, 9],
    "SMOKE100": [7, 9],
    "SMOKDAY2": [7, 9],
    "USENOW3": [7, 9],
    "ALCDAY5": [777, 888, 999],
    "AVEDRNK2": [77, 99],
    "DRNK3GE5": [77, 88, 99],
    "EXERANY2": [7, 9],
    # Extra Diabetes/Cardiovascular Questions
    "PREDIAB1": [7, 9],
    "DRADVISE": [7, 9],
    "HAREHAB1": [7, 9],
    "STREHAB1": [7, 9],
    "CVDASPRN": [7, 9],
    # Derived Intermediate variables
    "_RFHYPE5": [9],
    "_CHOLCHK": [9],
    "_RFCHOL": [9],
    "_AGEG5YR": [14],  # as 9
    "_AGE65YR": [3],  # as 9
    "_AGE80": [],
    "_AGE_G": [],
    "_BMI5": [],
    "_BMI5CAT": [],
    "_SMOKER3": [9],
    "_RFSMOK3": [9],
}

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------


def variable_columns():
    """
    Assigns each variable in the BRFSS dataset to a column index

    Returns:
        col_indices: dict, variables:col_idx

    """

    variables = "_STATE,FMONTH,IDATE,IMONTH,IDAY,IYEAR,DISPCODE,SEQNO,_PSU,CTELENUM,PVTRESD1,COLGHOUS,STATERES,CELLFON3,LADULT,NUMADULT,NUMMEN,NUMWOMEN,CTELNUM1,CELLFON2,CADULT,PVTRESD2,CCLGHOUS,CSTATE,LANDLINE,HHADULT,GENHLTH,PHYSHLTH,MENTHLTH,POORHLTH,HLTHPLN1,PERSDOC2,MEDCOST,CHECKUP1,BPHIGH4,BPMEDS,BLOODCHO,CHOLCHK,TOLDHI2,CVDSTRK3,ASTHMA3,ASTHNOW,CHCSCNCR,CHCOCNCR,CHCCOPD1,HAVARTH3,ADDEPEV2,CHCKIDNY,DIABETE3,DIABAGE2,SEX,MARITAL,EDUCA,RENTHOM1,NUMHHOL2,NUMPHON2,CPDEMO1,VETERAN3,EMPLOY1,CHILDREN,INCOME2,INTERNET,WEIGHT2,HEIGHT3,PREGNANT,QLACTLM2,USEEQUIP,BLIND,DECIDE,DIFFWALK,DIFFDRES,DIFFALON,SMOKE100,SMOKDAY2,STOPSMK2,LASTSMK2,USENOW3,ALCDAY5,AVEDRNK2,DRNK3GE5,MAXDRNKS,FRUITJU1,FRUIT1,FVBEANS,FVGREEN,FVORANG,VEGETAB1,EXERANY2,EXRACT11,EXEROFT1,EXERHMM1,EXRACT21,EXEROFT2,EXERHMM2,STRENGTH,LMTJOIN3,ARTHDIS2,ARTHSOCL,JOINPAIN,SEATBELT,FLUSHOT6,FLSHTMY2,IMFVPLAC,PNEUVAC3,HIVTST6,HIVTSTD3,WHRTST10,PDIABTST,PREDIAB1,INSULIN,BLDSUGAR,FEETCHK2,DOCTDIAB,CHKHEMO3,FEETCHK,EYEEXAM,DIABEYE,DIABEDU,CAREGIV1,CRGVREL1,CRGVLNG1,CRGVHRS1,CRGVPRB1,CRGVPERS,CRGVHOUS,CRGVMST2,CRGVEXPT,VIDFCLT2,VIREDIF3,VIPRFVS2,VINOCRE2,VIEYEXM2,VIINSUR2,VICTRCT4,VIGLUMA2,VIMACDG2,CIMEMLOS,CDHOUSE,CDASSIST,CDHELP,CDSOCIAL,CDDISCUS,WTCHSALT,LONGWTCH,DRADVISE,ASTHMAGE,ASATTACK,ASERVIST,ASDRVIST,ASRCHKUP,ASACTLIM,ASYMPTOM,ASNOSLEP,ASTHMED3,ASINHALR,HAREHAB1,STREHAB1,CVDASPRN,ASPUNSAF,RLIVPAIN,RDUCHART,RDUCSTRK,ARTTODAY,ARTHWGT,ARTHEXER,ARTHEDU,TETANUS,HPVADVC2,HPVADSHT,SHINGLE2,HADMAM,HOWLONG,HADPAP2,LASTPAP2,HPVTEST,HPLSTTST,HADHYST2,PROFEXAM,LENGEXAM,BLDSTOOL,LSTBLDS3,HADSIGM3,HADSGCO1,LASTSIG3,PCPSAAD2,PCPSADI1,PCPSARE1,PSATEST1,PSATIME,PCPSARS1,PCPSADE1,PCDMDECN,SCNTMNY1,SCNTMEL1,SCNTPAID,SCNTWRK1,SCNTLPAD,SCNTLWK1,SXORIENT,TRNSGNDR,RCSGENDR,RCSRLTN2,CASTHDX2,CASTHNO2,EMTSUPRT,LSATISFY,ADPLEASR,ADDOWN,ADSLEEP,ADENERGY,ADEAT1,ADFAIL,ADTHINK,ADMOVE,MISTMNT,ADANXEV,QSTVER,QSTLANG,MSCODE,_STSTR,_STRWT,_RAWRAKE,_WT2RAKE,_CHISPNC,_CRACE1,_CPRACE,_CLLCPWT,_DUALUSE,_DUALCOR,_LLCPWT,_RFHLTH,_HCVU651,_RFHYPE5,_CHOLCHK,_RFCHOL,_LTASTH1,_CASTHM1,_ASTHMS1,_DRDXAR1,_PRACE1,_MRACE1,_HISPANC,_RACE,_RACEG21,_RACEGR3,_RACE_G1,_AGEG5YR,_AGE65YR,_AGE80,_AGE_G,HTIN4,HTM4,WTKG3,_BMI5,_BMI5CAT,_RFBMI5,_CHLDCNT,_EDUCAG,_INCOMG,_SMOKER3,_RFSMOK3,DRNKANY5,DROCDY3_,_RFBING5,_DRNKWEK,_RFDRHV5,FTJUDA1_,FRUTDA1_,BEANDAY_,GRENDAY_,ORNGDAY_,VEGEDA1_,_MISFRTN,_MISVEGN,_FRTRESP,_VEGRESP,_FRUTSUM,_VEGESUM,_FRTLT1,_VEGLT1,_FRT16,_VEG23,_FRUITEX,_VEGETEX,_TOTINDA,METVL11_,METVL21_,MAXVO2_,FC60_,ACTIN11_,ACTIN21_,PADUR1_,PADUR2_,PAFREQ1_,PAFREQ2_,_MINAC11,_MINAC21,STRFREQ_,PAMISS1_,PAMIN11_,PAMIN21_,PA1MIN_,PAVIG11_,PAVIG21_,PA1VIGM_,_PACAT1,_PAINDX1,_PA150R2,_PA300R2,_PA30021,_PASTRNG,_PAREC1,_PASTAE1,_LMTACT1,_LMTWRK1,_LMTSCL1,_RFSEAT2,_RFSEAT3,_FLSHOT6,_PNEUMO2,_AIDTST3".split(
        ","
    )

    col_indices = {
        variable: i for i, variable in enumerate(variables) if variable in cvd_variables
    }

    return col_indices


def preprocess_cols(x_train, x_test, cols_indices):
    """
    Preprocesses each column of interest contained in cvd_variables for x_train and x_test
    Replaces zero codes by 0 and NaN codes by NaN in the columns of each cvd_variables key

    Parameters:
        x_train: numpy array of shape(N_1,D)
        x_test: numpy array of shape(N_2,D)
        cols_indices: dict
            Dictionary containing variable:col_idx pairs

    Returns:
        x_train: numpy array of shape(N_1,D)
            columns associated with cvd_variables have been treated depending on their codes
        x_test: numpy array of shape(N_2,D)
            columns associated with cvd_variables have been treated depending on their codes

    Notes:
    NaN codes are generally 7,77,777,7777,...; 9,99,999,9999,...; but not always
    Zero codes are generally 8,88,888,8888,...; but not always
    cvd_variables is a dict containing variable:[codes] for specific responses according to the BRFSS 2015 codebook
    cvd_variables and preproces_cols must be updated to account for new codes and new variables of interest

    """

    x_train = x_train.astype(float, copy=True)
    x_test = x_test.astype(float, copy=True)

    # Define cvd_variable zero and NaN codes
    zero_codes = [88, 888]
    nan_codes = [7, 77, 777, 7777, 9, 99, 999, 9999, 98, 3, 14]

    # Applies code to each variable in cvd_variables
    for variable, code in cvd_variables.items():
        if len(code) == 0:
            continue

        i = cols_indices[variable]

        # Determine code operations for each variable
        codes_to_zero = [c for c in code if c in zero_codes]
        codes_to_nan = [c for c in code if c in nan_codes]

        # Apply Preprocessing
        if codes_to_zero:
            x_train[np.isin(x_train[:, i], codes_to_zero), i] = 0
            x_test[np.isin(x_test[:, i], codes_to_zero), i] = 0
        if codes_to_nan:
            x_train[np.isin(x_train[:, i], codes_to_nan), i] = np.nan
            x_test[np.isin(x_test[:, i], codes_to_nan), i] = np.nan

    return x_train, x_test


def preprocessing_X(x_train, x_test):
    """
    Computes the design matrix X of x_train and x_test of shape (N_1,D) & (N_2,D) respectively
    This process keeps a subsample of dimensions belonging to cvd_variables, and
    for which at least one observation of x_train is non-nan and/or non-inf, whilst
    adding median value of a column to nan values in that column
    removing constant columns to allow normalization to control different values of regressors,
    normalizing,
    and adding a constant.
    Columns of interest are preselected using cvd_variables (via variable_columns())
    This is applied to both x_train and x_test to ensure equal dimensions D*,
    using median and standardization values of x_train

    Parameters:
        x_train: shape=(N_1,D)
        x_test: shape=(N_2,D)

    Returns:
        X_train: shape=(N_1,D*)
        X_test: shape=(N_2,D*)

    """

    # Keep cvd_variable columns
    valid_columns = variable_columns()
    valid_columns = np.array(list(valid_columns.values()))

    x_train_subsample = x_train[:, valid_columns]
    x_test_subsample = x_test[:, valid_columns]

    # Keep columns with at least one finite value
    nonnan_columns = np.where(np.isfinite(x_train_subsample).any(axis=0))[0]
    x_train_subsample = x_train_subsample[:, nonnan_columns]
    x_test_subsample = x_test_subsample[:, nonnan_columns]

    # Replace infinity with NaN
    x_train_subsample[~np.isfinite(x_train_subsample)] = np.nan
    x_test_subsample[~np.isfinite(x_test_subsample)] = np.nan

    # Replace NaN with test column median

    medians = np.nanmedian(x_train_subsample, axis=0)

    rows, cols = np.where(np.isnan(x_train_subsample))
    x_train_subsample[rows, cols] = medians[cols]

    rows, cols = np.where(np.isnan(x_test_subsample))
    x_test_subsample[rows, cols] = medians[cols]

    # Remove constant columns to prevent division by 0 during standardization
    nonconst_columns = np.where(np.std(x_train_subsample, axis=0) > 0)[0]
    x_train_subsample = x_train_subsample[:, nonconst_columns]
    x_test_subsample = x_test_subsample[:, nonconst_columns]

    # Standardization to prevent large gradient jumps due to extreme values
    mean = np.mean(x_train_subsample, axis=0)
    std = np.std(x_train_subsample, axis=0)

    x_train_subsample = (x_train_subsample - mean) / std
    x_test_subsample = (x_test_subsample - mean) / std

    # Form design matrix

    X_train = np.c_[np.ones((x_train.shape[0], 1)), x_train_subsample]
    X_test = np.c_[np.ones((x_test.shape[0], 1)), x_test_subsample]

    return X_train, X_test


def preprocessing_Y(y_train):
    """
    Converts y={-1,1} to y ={0,1}

    Parameters:
        y_train: numpy array of shape=(N,) with y={-1,1}

    Return:
        Y_train: numpy array of shape=(N,) with y={0,1}
    """

    if (y_train < 0).any():
        Y_train = (y_train == 1).astype(float)
    else:
        Y_train = y_train

    return Y_train


# ---------------------------------------------------------------------------
# 3. 3. Evaluation & Optimization
# ---------------------------------------------------------------------------


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def compute_f1score(y_test, y_pred):
    """
    Computes f_score

    Parameters:
        y_test: numpy array shape=(N,)
            y={0,1}
        y_pred: numpy array of shape=(N,)
            y={0,1}

    Returns:
        f1score: float
    """

    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return f1score


def tune_hyperparams(x_train, y_train, k_fold=2):
    """
    Performs cross validation to tune gamma (step-size), lambda(regularization param) and threshold(prediction threshold) hyperparameters
    Finds "best" parameters for maximizing f1 score

    Parameters:
        x_train: numpy array shape=(N,D)
            Training set
        y_train: numpy array shape=(N,)
            Training set y={-1,1}

    Returns:
        gamma: float
            "Best" step-size
        lambda_: float
            "Best" regularization hyperparameter
        threshold: float
            "Best" prediction threshold hyperparameter
        best_f1score: float

    """

    if k_fold < 2:
        raise ValueError("Number of folds must be larger than or equal to 2")

    # Set parameters ranges and initial values
    # lambdas = np.concatenate(([0.0],np.logspace(-6,0,20))) # Add lambda_ = 0 to account for classic non-regularized logistic regression
    # gammas = np.logspace(-3,0,20)
    # thresholds = np.arange(0.1,0.7,0.1)
    lambdas = [LAMBDA]
    gammas = [GAMMA]
    thresholds = [THRESHOLD]
    best_f1score = -1
    lambda_ = 0
    gamma = 0
    threshold = 0

    # Builds cv folds
    k_indices = build_k_indices(y_train, k_fold, SEED)

    # Cross validation
    for t in thresholds:

        for g in gammas:

            for l in lambdas:

                f1_scores = []

                for k in range(k_fold):

                    y_true_cv, y_pred_cv = cv_reg(
                        y_train, x_train, k_indices, l, k, g, t
                    )

                    f1_temp = compute_f1score(y_true_cv, y_pred_cv)

                    f1_scores.append(f1_temp)

                f1_mean = np.mean(f1_scores)
                if best_f1score < f1_mean:  # Sets "best" parameters

                    best_f1score = f1_mean
                    threshold = t
                    lambda_ = l
                    gamma = g

    return (
        gamma,
        lambda_,
        threshold,
        best_f1score,
    )


def cv_reg(y, x, k_indices, lambda_, k_fold, gamma, threshold):
    """
    Create folds on training data to create cross validation train and test sets
    Preprocesses each fold to ensure equal dimensions and specific normalization for each fold
    Calculates weights and predictions

    Parameters:
        x: numpy array shape=(N,D)
        y: numpy array shape=(N,)
            y={-1,1}
        k_indices: numpy array shape=(k_fold, N/k_fold)
            Indicates the data indices for each fold
        lambda_ : float
        k_fold: float
            tested fold
        gamma: float
        threshold: float

    Returns:
        y_test: numpy array shape=(N_test,)
            test outcomes y={0,1}
        y_pred: numpy array shape=(N_test,)
            prediction outcomes y={0,1}

    """

    # Assign test and train folds

    x_test_cr = x[k_indices[k_fold], :]
    y_test_cr = y[k_indices[k_fold]]

    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k_fold)].reshape(-1)

    x_train_cr = x[tr_indices, :]
    y_train_cr = y[tr_indices]

    # Preprocess each fold

    x_train_cr, x_test_cr = preprocessing_X(x_train_cr, x_test_cr)
    y_train_cr = preprocessing_Y(y_train_cr)
    y_test_cr = preprocessing_Y(y_test_cr)

    # Calculate weights on training set and predict using test set

    initial_w = np.zeros((x_train_cr.shape[1],))

    optimal_weights_cv, _ = reg_logistic_regression(
        y_train_cr, x_train_cr, lambda_, initial_w, MAX_ITERS, gamma
    )

    y_pred_cr = np.where(sigmoid(x_test_cr @ optimal_weights_cv) >= threshold, 1, 0)

    return y_test_cr, y_pred_cr


def optimization(X_train, Y_train, lambda_=0):
    """
    Perform optimization for standard or penalized logistic regression

    Parameters:
        X_train: numpy array shape=(N,D)
        Y_train: numpy array shape=(N,)
            y={0,1}
        lambda_: float
            regularization paramter, default = 0

    Returns:
        w: numpy array shape=(D,)
            optimized parameters
        loss : loss

    Notes:
    Default lambda_ = 0 will mean the function uses logistic_regression over reg_logistic_regression

    """

    w = np.zeros((X_train.shape[1],))

    if lambda_ == 0:
        w, loss = logistic_regression(Y_train, X_train, w, MAX_ITERS, GAMMA)

    else:
        w, loss = reg_logistic_regression(
            Y_train, X_train, lambda_, w, MAX_ITERS, GAMMA
        )

    return w, loss


def prediction(X_test, w):
    """
    Predict y values for a test set X_test using weights w.

    Parameters:
        X_train: numpy array shape=(N,D)
        w: numpy array shape=(D,)

    Returns:
        predictions: numpy array shape=(N,)
            y={-1,1}

    """

    predictions = sigmoid(X_test.dot(w))

    predictions = np.where(predictions >= THRESHOLD, 1, -1)

    return predictions


# ---------------------------------------------------------------------------
# 4. Main Execution Flow
# ---------------------------------------------------------------------------


def main():
    global GAMMA, LAMBDA, THRESHOLD, KFOLD

    print("Running main...")

    np.random.seed(SEED)

    print("1. Loading dataset...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    print("2. Preprocessing columns...")

    column_indices = variable_columns()
    x_train, x_test = preprocess_cols(x_train, x_test, cols_indices=column_indices)

    print("3. Fine tuning GAMMA, LAMBDA and THRESHOLD...")

    GAMMA, LAMBDA, THRESHOLD, f1score = tune_hyperparams(x_train, y_train, k_fold=KFOLD)

    print(
        "Best GAMMA: ",
        GAMMA,
        "Best LAMBDA: ",
        LAMBDA,
        "Best THRESHOLD: ",
        THRESHOLD,
        "Best KFOLD: ",
        KFOLD,
    )

    print(
        "CV f1score:",
        f1score,
        " for LAMBDA:",
        LAMBDA,
        "GAMMA:",
        GAMMA,
        "KFOLD:",
        KFOLD,
        "and THRESHOLD:",
        THRESHOLD,
    )

    print("4. Preprocessing dataset...")

    X_train, X_test = preprocessing_X(x_train, x_test)
    Y_train = preprocessing_Y(y_train)

    print("Diagnostic:Max |X|:", np.max(np.abs(X_train)))
    print("Features finite:", np.isfinite(X_train).all())
    print("Features finite:", np.isfinite(X_test).all())
    print("Correct Y:", np.min(Y_train))

    print("5. Optimization...")

    w, _ = optimization(X_train, Y_train, lambda_=LAMBDA)

    print("6. Predicting...")

    y_pred = prediction(X_test, w)

    print("Finite predictions:", np.isfinite(y_pred).all())

    print("7. Creating Submission")
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)

    print("END")


if __name__ == "__main__":
    main()
