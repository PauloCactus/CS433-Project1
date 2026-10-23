import numpy as np
import matplotlib.pyplot as plt
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
threshold = 1e-8
CROSS = True
PENALIZED = True
KFOLD = 4

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def associated_columns(x_train):
    variable_names = """_STATE FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU CTELENUM PVTRESD1 COLGHOUS STATERES CELLFON3 LADULT NUMADULT NUMMEN NUMWOMEN CTELNUM1 CELLFON2 CADULT PVTRESD2 CCLGHOUS CSTATE LANDLINE HHADULT GENHLTH PHYSHLTH MENTHLTH POORHLTH HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO CHOLCHK TOLDHI2 CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY DIABETE3 DIABAGE2 SEX MARITAL EDUCA RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 EMPLOY1 CHILDREN INCOME2 INTERNET WEIGHT2 HEIGHT3 PREGNANT QLACTLM2 USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON SMOKE100 SMOKDAY2 STOPSMK2 LASTSMK2 USENOW3 ALCDAY5 AVEDRNK2 DRNK3GE5 MAXDRNKS FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1 EXERANY2 EXRACT11 EXEROFT1 EXERHMM1 EXRACT21 EXEROFT2 EXERHMM2 STRENGTH LMTJOIN3 ARTHDIS2 ARTHSOCL JOINPAIN SEATBELT FLUSHOT6 FLSHTMY2 IMFVPLAC PNEUVAC3 HIVTST6 HIVTSTD3 WHRTST10 PDIABTST PREDIAB1 INSULIN BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK EYEEXAM DIABEYE DIABEDU CAREGIV1 CRGVREL1 CRGVLNG1 CRGVHRS1 CRGVPRB1 CRGVPERS CRGVHOUS CRGVMST2 CRGVEXPT VIDFCLT2 VIREDIF3 VIPRFVS2 VINOCRE2 VIEYEXM2 VIINSUR2 VICTRCT4 VIGLUMA2 VIMACDG2 CIMEMLOS CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT LONGWTCH DRADVISE ASTHMAGE ASATTACK ASERVIST ASDRVIST ASRCHKUP ASACTLIM ASYMPTOM ASNOSLEP ASTHMED3 ASINHALR HAREHAB1 STREHAB1 CVDASPRN ASPUNSAF RLIVPAIN RDUCHART RDUCSTRK ARTTODAY ARTHWGT ARTHEXER ARTHEDU TETANUS HPVADVC2 HPVADSHT SHINGLE2 HADMAM HOWLONG HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM BLDSTOOL LSTBLDS3 HADSIGM3 HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME PCPSARS1 PCPSADE1 PCDMDECN SCNTMNY1 SCNTMEL1 SCNTPAID SCNTWRK1 SCNTLPAD SCNTLWK1 SXORIENT TRNSGNDR RCSGENDR RCSRLTN2 CASTHDX2 CASTHNO2 EMTSUPRT LSATISFY ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE MISTMNT ADANXEV QSTVER QSTLANG MSCODE _STSTR _STRWT _RAWRAKE _WT2RAKE _CHISPNC _CRACE1 _CPRACE _CLLCPWT _DUALUSE _DUALCOR _LLCPWT _RFHLTH _HCVU651 _RFHYPE5 _CHOLCHK _RFCHOL _LTASTH1 _CASTHM1 _ASTHMS1 _DRDXAR1 _PRACE1 _MRACE1 _HISPANC _RACE _RACEG21 _RACEGR3 _RACE_G1 _AGEG5YR _AGE65YR _AGE80 _AGE_G HTIN4 HTM4 WTKG3 _BMI5 _BMI5CAT _RFBMI5 _CHLDCNT _EDUCAG _INCOMG _SMOKER3 _RFSMOK3 DRNKANY5 DROCDY3_ _RFBING5 _DRNKWEK _RFDRHV5 FTJUDA1_ FRUTDA1_ BEANDAY_ GRENDAY_ ORNGDAY_ VEGEDA1_ _MISFRTN _MISVEGN _FRTRESP _VEGRESP _FRUTSUM _VEGESUM _FRTLT1 _VEGLT1 _FRT16 _VEG23 _FRUITEX _VEGETEX _TOTINDA METVL11_ METVL21_ MAXVO2_ FC60_ ACTIN11_ ACTIN21_ PADUR1_ PADUR2_ PAFREQ1_ PAFREQ2_ _MINAC11 _MINAC21 STRFREQ_ PAMISS1_ PAMIN11_ PAMIN21_ PA1MIN_ PAVIG11_ PAVIG21_ PA1VIGM_ _PACAT1 _PAINDX1 _PA150R2 _PA300R2 _PA30021 _PASTRNG _PAREC1 _PASTAE1 _LMTACT1 _LMTWRK1 _LMTSCL1 _RFSEAT2 _RFSEAT3 _FLSHOT6 _PNEUMO2 _AIDTST3""".split()

    print("Number of names:", len(variable_names))

    valid_columns = np.where((np.isfinite(x_train).all(axis=0)) & (np.std(x_train,axis = 0)>0))[0]

    for i in valid_columns:
        print(i, variable_names[i])

def preprocessing_X(x_train, x_test):
    '''
    Computes the design matrix X of x_train 
    This process keeps a subsample of dimensions for which all observations of x_train are non-nan and/or non-inf,
    whilst removing dimensions with a low effect on the end result
    
    
    
    
    '''


    valid_columns = np.where((np.isfinite(x_train).all(axis=0)) & (np.std(x_train,axis = 0)>0))[0]

    #print("Valid Columns:", valid_columns[9:])

    x_train_subsample = x_train[:,valid_columns[9:]]
    x_test_subsample = x_test[:,valid_columns[9:]]


    # Normalize to prevent large numbers affecting gradient

    non_constant = np.where((np.std(x_train_subsample, axis = 0) > 0))[0]

    x_train_subsample = x_train_subsample[:,non_constant]
    x_test_subsample = x_test_subsample[:,non_constant]

    mean = np.mean(x_train_subsample, axis = 0)
    std = np.std(x_train_subsample, axis = 0)

    x_train_subsample = (x_train_subsample - mean)/std
    x_test_subsample = (x_test_subsample - mean)/std

    #Form classification matrix

    X_train = np.c_[np.ones((x_train.shape[0], 1)), x_train_subsample]
    X_test = np.c_[np.ones((x_test.shape[0], 1)), x_test_subsample]

    return X_train, X_test

def preprocessing_Y(y_train):
    '''
    Converts y={-1,1} to y ={0,1}
    
    
    '''
    if ((y_train <0).any()):
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

def cv_reg(y,x,k_indices,lambda_,k):

    initial_w= np.zeros((y.shape[0]))

    x_test_cr = x[k_indices[k],:]
    y_test_cr = y[k_indices[k],:]
                    
                    
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
                    
    x_train_cr = x[tr_indices,:]
    y_train_cr = y[tr_indices,:]

    optimal_weights_cv = reg_logistic_regression(y_train_cr,x_train_cr,lambda_,initial_w,MAX_ITERS,GAMMA)

    loss_te_cv = calculate_loss(y_test_cr,x_test_cr,optimal_weights_cv)

    return loss_te_cv

    

def optimization (X_train, Y_train, penalized= False, cross = False, k_fold=0):
    '''
    Perform optimization for standard or penalized logistic regression, with K-cross validation possible for penalized logistic regression
    
    
    
    '''
    loss = np.inf
    w = np.zeros((X_train.shape[1],))
    lambda_ = 0
    

    if penalized:

        lambdas = np.logspace(-4, 0, 10)

        if cross:

            best_loss = np.inf
            k_indices = build_k_indices(Y_train,k_fold,SEED)

            for l in lambdas:

                losses =[]

                for k in k_fold:

                    loss_te_cv = cv_reg(Y_train,X_train, k_indices,l,k)

                    losses.append(loss_te_cv)
                
                losses_mean = np.mean(losses)
                if losses_mean < best_loss:

                    best_loss = losses_mean
                    lambda_ = l

            w, loss = reg_logistic_regression(Y_train,X_train,l, w,MAX_ITERS,GAMMA)

        else: 

            for l in lambdas:

                w_pred, loss_pred = reg_logistic_regression(Y_train, X_train,l, w, MAX_ITERS, GAMMA)

                if loss_pred < loss:
                    loss=loss_pred
                    w = w_pred
                    lambda_=l
                


    else:
        w, loss = logistic_regression(Y_train, X_train, w, MAX_ITERS, GAMMA)

    return w, loss, lambda_


# ---------------------------------------------------------------------------
# 4. Main Execution Flow
# ---------------------------------------------------------------------------

def main():
    print('Running main...')

    np.random.seed(SEED)

    print("1. Loading dataset...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH,sub_sample=True)

    #print("Finding associated variables...")
    #associated_columns(x_train)

    
    print("2. Preprocessing dataset...")
    
    X_train, X_test = preprocessing_X(x_train,x_test)
    Y_train = preprocessing_Y(y_train)

    print("Diagnostic:Max |X|:", np.max(np.abs(X_train)))
    print("Features finite:", np.isfinite(X_train).all())

    print('Correct Y:', np.min(Y_train))

    print("3. Optimization ...")

    w, loss, lambda_ = optimization(X_train, Y_train)
    
    print("loss:", loss, " for lambda:", lambda_)
    # visualization
    #plt.plot(range(MAX_ITERS), losses)
    #print("loss={l}".format(l=calculate_loss(y_train, tx, w)))
    print("END")


if __name__ == "__main__":
    main()