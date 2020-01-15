import numpy as np
def eq_op_score(y_true, y_hat, sensitive_col, positive_target=1):

    if not np.all((sensitive_col == 0) | (sensitive_col == 1)):
        raise ValueError(
            f"equal_opportunity_score only supports binary indicator columns for `column`. "
            f"Found values {np.unique(sensitive_col)}"
        )

    y_given_z1_y1 = y_hat[(sensitive_col == 1) & (y_true == positive_target)]
    y_given_z0_y1 = y_hat[(sensitive_col == 0) & (y_true == positive_target)]

    # If we never predict a positive target for one of the subgroups, the model is by definition not
    # fair so we return 0
    if len(y_given_z1_y1) == 0:
        warnings.warn(
            f"No samples with y_hat == {positive_target} for {sensitive_column} == 1, returning 0",
            RuntimeWarning,
        )
        return 0

    if len(y_given_z0_y1) == 0:
        warnings.warn(
            f"No samples with y_hat == {positive_target} for {sensitive_column} == 0, returning 0",
            RuntimeWarning,
        ) 
        return 0

    p_y1_z1 = np.mean(y_given_z1_y1 == positive_target)
    p_y1_z0 = np.mean(y_given_z0_y1 == positive_target)
    score = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
    return score if not np.isnan(score) else 1

def dem_par_score(y_true, y_hat, sensitive_col, positive_target=1):
    if not np.all((sensitive_col == 0) | (sensitive_col == 1)):
        raise ValueError(
            f"p_percent_score only supports binary indicator columns for `column`. "
            f"Found values {np.unique(sensitive_col)}"
        )

    y_given_z1 = y_hat[sensitive_col == 1]
    y_given_z0 = y_hat[sensitive_col == 0]
    p_y1_z1 = np.mean(y_given_z1 == positive_target)
    p_y1_z0 = np.mean(y_given_z0 == positive_target)

    # If we never predict a positive target for one of the subgroups, the model is by definition not
    # fair so we return 0
    if p_y1_z1 == 0:
        warnings.warn(
            f"No samples with y_hat == {positive_target} for {sensitive_column} == 1, returning 0",
            RuntimeWarning,
        )
        return 0

    if p_y1_z0 == 0:
        warnings.warn(
            f"No samples with y_hat == {positive_target} for {sensitive_column} == 0, returning 0",
            RuntimeWarning,
        )
        return 0

    p_percent = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
    return p_percent if not np.isnan(p_percent) else 1
