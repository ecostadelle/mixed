# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import itertools
import numbers
from warnings import warn

import numpy as np
# from joblib import Parallel
from pandas import DataFrame
from scipy.stats import entropy

from sklearn.ensemble._bagging import _generate_indices, BaggingClassifier, MAX_INT
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state, indices_to_mask, check_array
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, _deprecate_positional_args, _check_sample_weight, \
    check_is_fitted


def generate_biased_indices(random_state, bootstrap, n_population, n_samples, probabilities):

    return random_state.choice(range(n_population), size=n_samples, replace=bootstrap, p=probabilities)


def generate_biased_bagging_indices(random_state, bootstrap_features,
                                    bootstrap_samples, n_features, n_samples,
                                    max_features, max_samples, sample_bias=None, feature_bias=None):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    if feature_bias is None:
        feature_indices = _generate_indices(random_state, bootstrap_features,
                                            n_features, max_features)
    else:
        feature_indices = generate_biased_indices(random_state, bootstrap_features,
                                                  n_features, max_features, feature_bias)
    if sample_bias is None:
        sample_indices = _generate_indices(random_state, bootstrap_samples,
                                           n_samples, max_samples)
    else:
        sample_indices = generate_biased_indices(random_state, bootstrap_samples,
                                                 n_samples, max_samples, sample_bias)

    return feature_indices, sample_indices


def parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                              seeds, total_n_estimators, verbose, sample_bias=None, feature_bias=None):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = generate_biased_bagging_indices(random_state,
                                                            bootstrap_features,
                                                            bootstrap, n_features,
                                                            n_samples, max_features,
                                                            max_samples,
                                                            sample_bias=sample_bias, feature_bias=feature_bias)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def parallel_compute_feature_importances(estimators, estimators_features, n_features, sufficiency_based=False, X=None):
    """Private function used to compute feature importances within a job."""
    importances = np.zeros((n_features,))

    for estimator, features in zip(estimators, estimators_features):
        if sufficiency_based and X is not None:
            importances[features] += estimator.sufficiency_based_feature_importances(X.iloc[:, features]
                                                                                     if isinstance(X, DataFrame)
                                                                                     else X[:, features])
        if not sufficiency_based and X is not None:
            importances[features] += estimator._dbcp(X.iloc[:, features]
                                                     if isinstance(X, DataFrame)
                                                     else X[:, features])
        else:
            importances[features] += estimator.feature_importances_

    return importances


def parallel_minimal_sufficient(estimators, estimators_features, n_features, X):
    """Private function used to compute feature importances within a job."""
    all_minimal_sufficient = None

    for estimator, features in zip(estimators, estimators_features):
        minimal_sufficient = np.full((X.shape[0], n_features), False)
        minimal_sufficient[:, features] = estimator.minimal_sufficient_features(X.iloc[:, features]
                                                                                if isinstance(X, DataFrame)
                                                                                else X[:, features])
        all_minimal_sufficient = minimal_sufficient if all_minimal_sufficient is None else np.append(all_minimal_sufficient, minimal_sufficient, axis=0)

    return all_minimal_sufficient


def compute_sample_bias(x, uncertain_features):
    bias_sum = 0.0
    sample_bias = np.empty((x.shape[0],), dtype=np.float64)
    for s in range(x.shape[0]):
        entropy_sum = 0.0
        missing_values = 0
        for f in range(x.shape[1]):
            if uncertain_features[f]:
                v = x.iloc[s, f] if isinstance(x, DataFrame) else x[s, f]
                if v > 0:
                    entropy_sum += entropy([v, 1.0 - v])
                else:
                    missing_values += 1
        if x.shape[1] == missing_values:
            sample_bias[s] = 0.0001
        else:
            sample_bias[s] = 1.0 - entropy_sum / (x.shape[1] - missing_values)
            sample_bias[s] *= (x.shape[1] - missing_values) / x.shape[1]    # Known values rate
        bias_sum += sample_bias[s]

    return sample_bias / bias_sum


def compute_feature_bias(x, uncertain_features, sample_indices=None):
    if sample_indices is None:
        sample_indices = range(x.shape[0])

    bias_sum = 0.0
    feature_bias = np.empty((x.shape[1],), dtype=np.float64)
    for f in range(x.shape[1]):
        entropy_sum = 0.0
        missing_values = 0
        if uncertain_features[f]:
            for s in sample_indices:
                v = x.iloc[s, f] if isinstance(x, DataFrame) else x[s, f]
                if v > 0:
                    entropy_sum += entropy([v, 1.0 - v])
                else:
                    missing_values += 1
        nb_indices = sample_indices.stop if isinstance(sample_indices, range) else sample_indices.shape[0]
        if nb_indices == missing_values:
            feature_bias[f] = 0.0001
        else:
            feature_bias[f] = 1.0 - entropy_sum / (nb_indices - missing_values)
            feature_bias[f] *= (nb_indices - missing_values) / nb_indices    # Known values rate
        bias_sum += feature_bias[f]

    return feature_bias / bias_sum


def my_parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "my_predict_proba"):
            proba_estimator = estimator.my_predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += \
                    proba_estimator[:, range(len(estimator.classes_))]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba

def balanced_weight_vector(y: np.ndarray) -> np.ndarray:
    error = ValueError("y must be a 1D numpy array")
    try:
        if y.ndim == 1:
            classes, freq = np.unique(y, return_counts=True)
            maj_index = freq.argmax()
            maj_class = classes[maj_index]
            maj_weight = freq.min()/freq.max()
            min_weight = 1
            return np.array([maj_weight if y_j == maj_class else min_weight for y_j in y])
        else:
            raise error
    
    except:
        raise error

class InterpretableBaggingClassifier(BaggingClassifier):

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_importances = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                   **self._parallel_args())(
            delayed(parallel_compute_feature_importances)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                self.n_features_in_)
            for i in range(n_jobs))

        # Reduce
        importances = sum(all_importances) / self.n_estimators

        return importances / np.sum(importances)

    def _dbcp(self, X):
        check_is_fitted(self)

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_importances = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                   **self._parallel_args())(
            delayed(parallel_compute_feature_importances)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                self.n_features_in_,
                sufficiency_based=False,
                X=X)
            for i in range(n_jobs))

        # Reduce
        importances = sum(all_importances) / self.n_estimators

        return importances

    def sufficiency_based_feature_importances(self, X):
        check_is_fitted(self)

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_importances = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                   **self._parallel_args())(
            delayed(parallel_compute_feature_importances)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                self.n_features_in_,
                sufficiency_based=True,
                X=X)
            for i in range(n_jobs))

        # Reduce
        importances = sum(all_importances) / self.n_estimators

        return importances / np.sum(importances)

    def include_attribute_weights(self, feature_weights=None):
        """ Public function used to incorporate attribute weights in the conditional probability."""
        if feature_weights is not None:
            for estimator, features in zip(self.estimators_, self.estimators_features_):
                estimator.include_attribute_weights_(feature_weights[features])
        else:
            raise ValueError("The attribute weights is not available for inclusion")
                
    def include_weights(self, feature_weights=None):
        """ Public function used to incorporate attribute weights in the conditional probability."""
        if feature_weights is not None:
            for estimator, features in zip(self.estimators_, self.estimators_features_):
                estimator.weights = (feature_weights[features])
        else:
            raise ValueError("The attribute weights is not available for inclusion")
                
    def backup_log_proba(self, restore=False):
        """ usage *.backup_log_proba() or *.backup_log_proba(restore=True) """
        check_is_fitted(self)
        for estimator in self.estimators_:
            estimator.backup_log_proba_(restore=restore)
        

    def my_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )

        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_in_, X.shape[1]))

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(my_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

class UDBaggingClassifier(InterpretableBaggingClassifier):
    @_deprecate_positional_args
    def __init__(self,
                 feature_bias=None,
                 estimator=None,
                 n_estimators=10, *,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 uncertain_features=None,
                 biased_bootstrap=False,
                 biased_subspaces=False,
                 default_feature_uncertainty=False):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.uncertain_features = uncertain_features
        self.biased_bootstrap = biased_bootstrap
        self.biased_subspaces = biased_subspaces
        self.default_feature_uncertainty = default_feature_uncertainty
        self.feature_bias=feature_bias

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False, multi_output=True
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_in_
        elif self.max_features == "sqrt":
            max_features = np.sqrt(self.n_features_in_)
        elif self.max_features == "log2":
            max_features = np.log2(self.n_features_in_)
        else:
            max_features = self.n_features_in_

        # Store validated integer feature sampling value
        self._max_features = max(1, int(max_features))

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        if self.uncertain_features is None:
            self.uncertain_features = np.full((self.n_features_in_,), self.default_feature_uncertainty)
        elif self.n_features_in_ > self.uncertain_features.shape[0]:
            self.uncertain_features = np.append(self.uncertain_features, np.full(
                                                                (self.n_features_in_ - self.uncertain_features.shape[0],),
                                                                self.default_feature_uncertainty))

        sample_bias = None
        if self.bootstrap and self.biased_bootstrap:
            sample_bias = compute_sample_bias(X, self.uncertain_features)

        feature_bias=self.feature_bias
        if self.biased_subspaces:
            if feature_bias is None:
                feature_bias = compute_feature_bias(X, self.uncertain_features)
                self.feature_bias = feature_bias

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               **self._parallel_args())(
            delayed(parallel_build_estimators)(
                n_estimators=n_estimators[i],
                ensemble=self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                seeds=seeds[starts[i]:starts[i + 1]],
                total_n_estimators=total_n_estimators,
                verbose=self.verbose,
                sample_bias=sample_bias, 
                feature_bias=feature_bias)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
