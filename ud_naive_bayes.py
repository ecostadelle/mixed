# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB, BernoulliNB, _BaseDiscreteNB, CategoricalNB, MultinomialNB
#from sklearn.utils import _deprecate_positional_args
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted


def binarize(X, threshold=0.0, missing_values=None):
    cond = X > threshold
    not_cond = X <= threshold
    if missing_values is not None:
        cond &= X != missing_values
        not_cond &= X != missing_values
    X[cond] = 1
    X[not_cond] = 0
    return X


class UDGaussianNB(GaussianNB):

    def __init__(self, *, priors=None, var_smoothing=1e-9, missing_values=-1.0):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.missing_values = missing_values

    def _update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        mx = np.ma.masked_equal(X, self.missing_values)

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(mx, axis=0, weights=sample_weight)
            new_var = np.average((mx - new_mu) ** 2, axis=0,
                                 weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(mx, axis=0)
            new_mu = np.mean(mx, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        mx = np.ma.masked_equal(X, self.missing_values)
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((mx - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


class UDBernoulliNB(_BaseDiscreteNB):

#    @_deprecate_positional_args
    def __init__(self, *, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None, missing_values=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.missing_values = missing_values

    def _check_X(self, X):
        X = super()._check_X(X)
        if self.missing_values is not None:
            X[X == self.missing_values] = -1
        if self.binarize is not None:
            if self.missing_values is not None:
                X = binarize(X, threshold=self.binarize, missing_values=-1)
            else:
                X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if self.missing_values is not None:
            X[X == self.missing_values] = -1
        if self.binarize is not None:
            if self.missing_values is not None:
                X = binarize(X, threshold=self.binarize, missing_values=-1)
            else:
                X = binarize(X, threshold=self.binarize)
        return X, y

    def _init_counters(self, n_effective_classes, n_features):
        super()._init_counters(n_effective_classes, n_features)
        self.neg_feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        mx = np.ma.masked_equal(X, -1)
        self.feature_count_ += np.ma.dot(Y.T, mx)
        self.class_count_ += Y.sum(axis=0)

        neg_mx = 1 - mx
        self.neg_feature_count_ += np.ma.dot(Y.T, neg_mx)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

        smoothed_neg_fc = self.neg_feature_count_ + alpha

        self.neg_feature_log_prob_ = (np.log(smoothed_neg_fc) -
                                      np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        mx = np.ma.masked_equal(X, -1)
        jll = np.ma.dot(mx, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        if self.missing_values is not None:
            neg_neg_prob = np.log(1 - np.exp(self.neg_feature_log_prob_))
            neg_mx = 1 - mx
            jll += np.ma.dot(neg_mx, (self.neg_feature_log_prob_ - neg_neg_prob).T)
            jll += neg_neg_prob.sum(axis=1)

        return jll.filled()


class InterpretableBernoulliNB(BernoulliNB):

#    @_deprecate_positional_args
    def __init__(self, *, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None, missing_values=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.missing_values = missing_values
        self.backup = None
        self.weights = None

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        return self.compute_feature_importances_()

    def compute_feature_importances_(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        feature_prob = np.exp(self.feature_log_prob_)
        importances = np.abs(feature_prob[0] - feature_prob[1])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def sufficiency_based_feature_importances(self, X, normalize=True):
        """Computes the importance of each feature (aka variable) based on its `sufficiency` for the examples in X."""
        X = self._check_X(X)
        minimal_sufficient = self.minimal_sufficient_features(X)
        importances = np.count_nonzero(minimal_sufficient, axis=0) / X.shape[0]
        # cardinalities = np.count_nonzero(minimal_sufficient, axis=1)
        # unique, indices, counts = np.unique(minimal_sufficient, axis=0, return_index=True, return_counts=True)
        # print('Inst\tFreq\tCard')
        # for i in range(unique.shape[0]):
        #     print(indices[i], '\t', counts[i], '\t', cardinalities[indices[i]])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def minimal_sufficient_features(self, X):
        """Returns a `minimal sufficient` set of features for each example in X."""
        X = self._check_X(X)
        y = self.predict(X)
        minimal_sufficient = self.supporting_features(X, y)
        sorted_indices = np.argsort(self.feature_importances_)
        for i in range(X.shape[0]):
            for j in sorted_indices:
                if minimal_sufficient[i, j]:
                    minimal_sufficient[i, j] = False
                    jll = self.reduced_joint_log_likelihood(X[i], minimal_sufficient[i])
                    if self.classes_[np.argmax(jll)] != y[i]:
                        minimal_sufficient[i, j] = True
                        break

        return minimal_sufficient

    def supporting_features(self, X, y):
        """Returns the `supporting` features for each example in X, given the corresponding classes in y."""
        support_features = np.empty(X.shape, dtype=np.bool)
        feature_prob = np.exp(self.feature_log_prob_)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] == 1:
                    support_features[i, j] = feature_prob[y[i], j] > feature_prob[1 - y[i], j]
                else:
                    support_features[i, j] = 1 - feature_prob[y[i], j] > 1 - feature_prob[1 - y[i], j]

        return support_features

    def reduced_joint_log_likelihood(self, x, mask):
        """Calculate the posterior log probability of the sample x considering a reduced model with the features
        filtered by an index mask """
        feature_log_prob = self.feature_log_prob_[:, mask]
        x = x[mask]

        neg_prob = np.log(1 - np.exp(feature_log_prob))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(x, (feature_log_prob - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll

    def include_attribute_weights_(self, feature_weights:np.array=None):
        if feature_weights is not None:
            pos_prob = np.exp(self.feature_log_prob_)
            neg_prob = 1 - pos_prob
            prob = np.stack((neg_prob, pos_prob), axis=1)
            prob = np.copy(np.exp(np.log(prob) * np.exp(-feature_weights)))
            self.feature_log_prob_ = np.log(prob[:,1,:]/prob.sum(axis=1))

    def backup_log_proba_(self, restore=False):
        if restore:
            if self.backup is not None:
                self.feature_log_prob_ = np.copy(self.backup)
            else:
                raise ValueError("No backup found for restoration")
        else:
            if self.backup is None:
                self.backup = np.copy(self.feature_log_prob_)
            else:
                raise ValueError("The backup has already been saved.")

    def my_joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        
        if self.weights is None:
            self.weights = np.zeros(n_features)
            
        weights = np.exp(-self.weights)
        
        jll = safe_sparse_dot(X, (self.feature_log_prob_ * weights).T)
        jll += safe_sparse_dot(1-X, (neg_prob * weights).T)
        jll += self.class_log_prior_

        return jll

    def my_predict_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        jll = self.my_joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return np.exp(jll - np.atleast_2d(log_prob_x).T)

class InterpretableMultinomialNB(MultinomialNB):

#    @_deprecate_positional_args
    def __init__(self, *, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        return self.compute_feature_importances_()

    def compute_feature_importances_(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        feature_prob = np.exp(self.feature_log_prob_)
        importances = np.abs(feature_prob[0] - feature_prob[1])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def sufficiency_based_feature_importances(self, X, normalize=True):
        """Computes the importance of each feature (aka variable) based on its `sufficiency` for the examples in X."""
        X = self._check_X(X)
        minimal_sufficient = self.minimal_sufficient_features(X)
        importances = np.count_nonzero(minimal_sufficient, axis=0) / X.shape[0]
        # cardinalities = np.count_nonzero(minimal_sufficient, axis=1)
        # unique, indices, counts = np.unique(minimal_sufficient, axis=0, return_index=True, return_counts=True)
        # print('Inst\tFreq\tCard')
        # for i in range(unique.shape[0]):
        #     print(indices[i], '\t', counts[i], '\t', cardinalities[indices[i]])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def minimal_sufficient_features(self, X):
        """Returns a `minimal sufficient` set of features for each example in X."""
        X = self._check_X(X)
        y = self.predict(X)
        minimal_sufficient = self.supporting_features(X, y)
        sorted_indices = np.argsort(self.feature_importances_)
        for i in range(X.shape[0]):
            for j in sorted_indices:
                if minimal_sufficient[i, j]:
                    minimal_sufficient[i, j] = False
                    jll = self.reduced_joint_log_likelihood(X[i], minimal_sufficient[i])
                    if self.classes_[np.argmax(jll)] != y[i]:
                        minimal_sufficient[i, j] = True
                        break

        return minimal_sufficient

    def supporting_features(self, X, y):
        """Returns the `supporting` features for each example in X, given the corresponding classes in y."""
        support_features = np.empty(X.shape, dtype=np.bool)
        feature_prob = np.exp(self.feature_log_prob_)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] == 1:
                    support_features[i, j] = feature_prob[y[i], j] > feature_prob[1 - y[i], j]
                else:
                    support_features[i, j] = 1 - feature_prob[y[i], j] > 1 - feature_prob[1 - y[i], j]

        return support_features

    def reduced_joint_log_likelihood(self, x, mask):
        """Calculate the posterior log probability of the sample x considering a reduced model with the features
        filtered by an index mask """
        feature_log_prob = self.feature_log_prob_[:, mask]
        x = x[mask]

        neg_prob = np.log(1 - np.exp(feature_log_prob))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(x, (feature_log_prob - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll
    
    
class InterpretableCategoricalNB(CategoricalNB):

#    @_deprecate_positional_args
    def __init__(self, *, alpha=1.0, fit_prior=True, class_prior=None, min_categories=None, force_alpha=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories
        self.force_alpha = force_alpha

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        return self.compute_feature_importances_()

    def compute_feature_importances_(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        q = len(self.classes_)
        m = self.n_features_in_
        importances = np.zeros(m)
        for j,f in enumerate(self.feature_log_prob_):
            p = np.exp(f)
            for r in range(q):
                for s in range(r+1, q):
                    importances[j] += np.abs(p[r] - p[s]).sum()
        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def sufficiency_based_feature_importances(self, X, normalize=True):
        """Computes the importance of each feature (aka variable) based on its `sufficiency` for the examples in X."""
        X = self._check_X(X)
        minimal_sufficient = self.minimal_sufficient_features(X)
        importances = np.count_nonzero(minimal_sufficient, axis=0) / X.shape[0]
        # cardinalities = np.count_nonzero(minimal_sufficient, axis=1)
        # unique, indices, counts = np.unique(minimal_sufficient, axis=0, return_index=True, return_counts=True)
        # print('Inst\tFreq\tCard')
        # for i in range(unique.shape[0]):
        #     print(indices[i], '\t', counts[i], '\t', cardinalities[indices[i]])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def minimal_sufficient_features(self, X):
        """Returns a `minimal sufficient` set of features for each example in X."""
        X = self._check_X(X)
        y = self.predict(X)
        minimal_sufficient = self.supporting_features(X, y)
        sorted_indices = np.argsort(self._dbcp(X))
        for i in range(X.shape[0]):
            for j in sorted_indices:
                if minimal_sufficient[i, j]:
                    minimal_sufficient[i, j] = False
                    jll = self.reduced_joint_log_likelihood(X[i], minimal_sufficient[i])
                    if self.classes_[np.argmax(jll)] != y[i]:
                        minimal_sufficient[i, j] = True
                        break

        return minimal_sufficient

    def supporting_features(self, X, y):
        """Returns the `supporting` features for each example in X, given the corresponding classes in y."""
        support_features = np.empty(X.shape, dtype=np.bool)
        n,m = X.shape
        for i in range(n):
            for j in range(m):
                # MISSINGVALUES - compute supporting feature just for valid categories
                if (X[i, j] >= 0) & (X[i, j] < self.n_categories_[j]):
                    support_features[i, j] = np.argmax(self.feature_log_prob_[j][:,X[i, j]]) == y[i]
            else:
                support_features[i, j] = False

        return support_features
    
    def _dbcp(self, X, normalize=False):
        """
        The DBCP method calculates Difference Between Conditional 
        Probabilities. It is a way to calculate the feature importances 
        for each x_ij, based on its category.
        """
        n,m = X.shape
        if isinstance(X, pd.DataFrame):
            X = X.values
        importances = np.zeros(m)
        for j in range(m):
            Xj = X[:, j].astype(int)
            p = np.exp(self.feature_log_prob_[j])
            # MISSINGVALUES - compute importance just for valid categories
            Xj = Xj[(Xj >= 0) & (Xj < self.n_categories_[j])]
            importances[j] = (np.abs(p[0] - p[1])[Xj]).sum()
        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def _joint_log_likelihood(self, X):
        self._check_n_features(X, reset=False)
        q = len(self.classes_)
        n,m = X.shape
        jll = np.zeros((n, q))
        for j in range(m):
            Xj = X[:, j].astype(int)
            # MISSINGVALUES - mask to preserve just valid values
            valid_categories = Xj[Xj >= 0]
            # MISSINGVALUES - pass if the feature likehood cant be calculated
            try:
                jll += self.feature_log_prob_[j][:, valid_categories].T
            except:
                pass
        jll += self.class_log_prior_
        return jll

    def reduced_joint_log_likelihood(self, X, mask):
        """Calculate the posterior log probability of the sample x considering a reduced model with the features
        filtered by an index mask """
        feature_log_prob = [x for i,x in enumerate(self.feature_log_prob_) if mask[i]]
        X = X[mask]
        
        m = len(X)

        jll = np.zeros(self.class_count_.shape[0])
        for j in range(m):
            # MISSINGVALUES - compute jll just for valid categories
            Xj = X[j]
            if (Xj >= 0) & (Xj < self.n_categories_[j]):
                try:
                    jll += feature_log_prob[j][:, Xj]
                except:
                    pass
            jll += self.class_log_prior_
        return jll


    # MISSINGVALUES - changed parameters: dtype=None, force_all_finite='allow-nan'
    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = self._validate_data(
            X, dtype=None, accept_sparse=False, force_all_finite='allow-nan', reset=False
        )
        # MISSINGVALUES pandas.CategoricalDtype encodes np.nan as -1
        # check_non_negative(X, "CategoricalNB (input X)")
        return X

    # MISSINGVALUES - changed parameters: dtype=None, force_all_finite='allow-nan'
    def _check_X_y(self, X, y, reset=True):
        X, y = self._validate_data(
            X, y, dtype=None, accept_sparse=False, force_all_finite='allow-nan', reset=reset
        )
        # MISSINGVALUES pandas.CategoricalDtype encodes np.nan as -1
        # check_non_negative(X, "CategoricalNB (input X)")
        return X, y

    def _init_counters(self, n_classes, n_features):
        # MISSINGVALUES - change class_count_ into a 2D array
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]

    @staticmethod
    def _validate_n_categories(X, min_categories):
        # rely on max for n_categories categories are encoded between 0...n-1
        # MISSINGVALUES - changed np.max to np.nanmax
        n_categories_X = np.nanmax(X, axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(
                    "'min_categories' should have integral type. Got "
                    f"{min_categories_.dtype} instead."
                )
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(
                    f"'min_categories' should have shape ({X.shape[1]},"
                    ") when an array-like is provided. Got"
                    f" {min_categories_.shape} instead."
                )
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        def _update_cat_count_dims(cat_count, highest_feature):
            # MISSINGVALUES - cast to int
            diff = int(highest_feature + 1 - cat_count.shape[1])
            if diff > 0:
                # we append a column full of zeros for each new category
                return np.pad(cat_count, [(0, 0), (0, diff)], "constant")
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                # MISSINGVALUES - mask to preserve just valid values
                mask &= ~np.isnan(X_feature)
                mask &= X_feature >= 0
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                # MISSINGVALUES - cast to int
                counts = np.bincount(X_feature[mask].astype('int'), weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            self.category_count_[i] = _update_cat_count_dims(
                self.category_count_[i], self.n_categories_[i] - 1
            )
            _update_cat_count(
                X_feature, Y, self.category_count_[i], self.class_count_.shape[0]
            )
