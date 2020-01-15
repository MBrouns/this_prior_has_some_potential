import pymc3 as pm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from metrics import eq_op_score, dem_par_score

def run_it(model, df):
    trace, predictor = model(df)

    y_hat_proba = predictor(trace, df)
    y_hat = y_hat_proba > 0.5
    pm.traceplot(trace);
    
    print('accuracy:', accuracy_score(df['released'], y_hat))
    print('precision:', precision_score(df['released'], y_hat))
    print('recall:', recall_score(df['released'], y_hat))
    
          
    print('eq op score:', eq_op_score(df['released'], y_hat,  df['colour']))
    print('dem par score:', dem_par_score(df['released'], y_hat, df['colour']))
    return trace, predictor