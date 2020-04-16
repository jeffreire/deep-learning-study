import Working_with_matrices_in_SVD as cod_svd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets as dt
from sklearn.decomposition import TruncatedSVD

X, y = dt.load_digits(return_X_y= True)
print(X.shape)
print(y.shape)
rf_ori = RandomForestClassifier(oob_score= True)
rf_ori.fit(X, y)
rf_ori.oob_score_

svd = TruncatedSVD(n_components = 16) 
X_reduced = svd.fit_transform(X)
print(svd.explained_variance_ratio_.sum())

rf_reduced = RandomForestClassifier(oob_score = True) 
rf_reduced.fit(X_reduced, y) 
print(rf_reduced.oob_score_)

