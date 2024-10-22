import pandas as pd
import  numpy as np
from numba import jit

def drop_corr_features_target(data_features: pd.DataFrame,
                              data_corr: pd.Series,
                              threshold_corr: int = 0.7
                              ) -> list:
    columns = data_features.columns
    corr_mtx = abs(pd.DataFrame(
        np.corrcoef(data_features.values, rowvar=False),
        index=columns, columns=columns)
        )
    avg_corr = corr_mtx.mean(axis=1)
    upper_tri = corr_mtx.where(
        np.triu(np.ones(corr_mtx.shape), k=1).astype(bool)
        )

    upper_tri_matrix = upper_tri.to_numpy()
    data_corr_values = data_corr.to_numpy()
    avg_corr_values = avg_corr.to_numpy()
    features = np.array(upper_tri.columns.to_list())

    features_to_drop = drop_features_numba(upper_tri_matrix,
                                           data_corr_values,
                                           avg_corr_values,
                                           features, 
                                           threshold_corr)

    dropcols_names = list(set(features_to_drop))

    return dropcols_names

@jit(nopython=True)
def drop_features_numba(upper_tri_matrix: np.array,
                        data_corr_values: np.array,
                        avg_corr_values: np.array,
                        features: np.array,
                        threshold_corr: int):

    features_to_drop = []
    for row in range(len(upper_tri_matrix) - 1):
        for col in range(row + 1, len(upper_tri_matrix)):
            if upper_tri_matrix[row, col] > threshold_corr:

                if data_corr_values[row] < data_corr_values[col]:
                    features_to_drop.append(features[row])
                elif data_corr_values[row] > data_corr_values[col]:
                    features_to_drop.append(features[col])
                else:
                    if avg_corr_values[row] > avg_corr_values[col]:
                        features_to_drop.append(features[col])
                    else:
                        features_to_drop.append(features[row])

    return features_to_drop
