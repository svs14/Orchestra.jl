# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

importall Orchestra.AbstractLearner
using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM
@pyimport sklearn.neighbors as NN
@pyimport sklearn.svm as SVM
@pyimport sklearn.tree as TREE

export SKLLearner,
       SKLRandomForest,
       SKLExtraTrees,
       SKLGradientBoosting,
       SKLLogisticRegression,
       SKLPassiveAggressive,
       SKLRidge,
       SKLRidgeCV,
       SKLSGD,
       SKLKNeighbors,
       SKLRadiusNeighbors,
       SKLNearestCentroid,
       SKLSVC,
       SKLLinearSVC,
       SKLNuSVC,
       SKLDecisionTree,
       train!,
       predict!

# SKLLearners have field 'model' of which
# corresponds to a Python BaseEstimator in scikit-learn.
abstract SKLLearner <: Learner

macro build_train!_func(orchestra_name, scikit_learner_name)
  @eval begin
    function train!(on::$orchestra_name, instances::Matrix, labels::Vector)
      impl_options = on.options[:impl_options]
      on.model = $scikit_learner_name(;impl_options...)
      on.model[:fit](instances, labels)
    end
  end
end

function predict!(skll::SKLLearner, instances::Matrix)
  return collect(skll.model[:predict](instances))
end


# Random forest.
# 
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Number of trees in forest.
#     :n_estimators => 10,
#     # Function to measure quality of a split (tree-specific).
#     # ("gini", "entropy")
#     :criterion => "gini",
#     # Number of features to consider when looking for the best split.
#     # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
#     :max_features => "auto",
#     # Maximum depth of the tree.
#     # (Int, nothing)
#     :max_depth => nothing,
#     # The minimum number of samples required to split 
#     # an internal node (tree-specific).
#     :min_samples_split => 2,
#     # The minimum number of samples in newly created leaves (tree-specific).
#     :min_samples_leaf => 1,
#     # Whether bootstrap samples are used when building trees.
#     :bootstrap => true,
#     # Whether to use out-of-bag samples to 
#     # estimate the generalization error.
#     :oob_score => true,
#     # The number of jobs to run in parallel for both fit and predict. 
#     # If -1, then the number of jobs is set to the number of cores.
#     :n_jobs => 1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing,
#     # Controls the verbosity of the tree building process.
#     :verbose => 0
#   },
# }
# </pre>
type SKLRandomForest <: SKLLearner
  model
  options
  
  function SKLRandomForest(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Number of trees in forest.
        :n_estimators => 10,
        # Function to measure quality of a split (tree-specific).
        # ("gini", "entropy")
        :criterion => "gini",
        # Number of features to consider when looking for the best split.
        # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
        :max_features => "auto",
        # Maximum depth of the tree.
        # (Int, nothing)
        :max_depth => nothing,
        # The minimum number of samples required to split 
        # an internal node (tree-specific).
        :min_samples_split => 2,
        # The minimum number of samples in newly created leaves (tree-specific).
        :min_samples_leaf => 1,
        # Whether bootstrap samples are used when building trees.
        :bootstrap => true,
        # Whether to use out-of-bag samples to 
        # estimate the generalization error.
        :oob_score => true,
        # The number of jobs to run in parallel for both fit and predict. 
        # If -1, then the number of jobs is set to the number of cores.
        :n_jobs => 1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # Controls the verbosity of the tree building process.
        :verbose => 0
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLRandomForest ENS.RandomForestClassifier


# Extra-trees.
# 
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Number of trees in forest.
#     :n_estimators => 10,
#     # Function to measure quality of a split (tree-specific).
#     # ("gini", "entropy")
#     :criterion => "gini",
#     # Number of features to consider when looking for the best split.
#     # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
#     :max_features => "auto",
#     # Maximum depth of the tree.
#     # (Int, nothing)
#     :max_depth => nothing,
#     # The minimum number of samples required to split 
#     # an internal node (tree-specific).
#     :min_samples_split => 2,
#     # The minimum number of samples in newly created leaves (tree-specific).
#     :min_samples_leaf => 1,
#     # Whether bootstrap samples are used when building trees.
#     :bootstrap => true,
#     # Whether to use out-of-bag samples to 
#     # estimate the generalization error.
#     :oob_score => true,
#     # The number of jobs to run in parallel for both fit and predict. 
#     # If -1, then the number of jobs is set to the number of cores.
#     :n_jobs => 1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing,
#     # Controls the verbosity of the tree building process.
#     :verbose => 0
#   },
# }
# </pre>
type SKLExtraTrees <: SKLLearner
  model
  options
  
  function SKLExtraTrees(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Number of trees in forest.
        :n_estimators => 10,
        # Function to measure quality of a split (tree-specific).
        # ("gini", "entropy")
        :criterion => "gini",
        # Number of features to consider when looking for the best split.
        # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
        :max_features => "auto",
        # Maximum depth of the tree.
        # (Int, nothing)
        :max_depth => nothing,
        # The minimum number of samples required to split 
        # an internal node (tree-specific).
        :min_samples_split => 2,
        # The minimum number of samples in newly created leaves (tree-specific).
        :min_samples_leaf => 1,
        # Whether bootstrap samples are used when building trees.
        :bootstrap => true,
        # Whether to use out-of-bag samples to 
        # estimate the generalization error.
        :oob_score => true,
        # The number of jobs to run in parallel for both fit and predict. 
        # If -1, then the number of jobs is set to the number of cores.
        :n_jobs => 1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # Controls the verbosity of the tree building process.
        :verbose => 0
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLExtraTrees ENS.ExtraTreesClassifier


# Gradient boosting machine.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Loss function to be optimized. 
#     # ("deviance")
#     :loss => "deviance",
#     # Learning rate shrinks the contribution of each tree by learning_rate.
#     :learning_rate => 0.1,
#     # The number of boosting stages to perform.
#     :n_estimators => 100,
#     # Maximum depth of the individual regression estimators. 
#     :max_depth => 3,
#     # The minimum number of samples required to split an internal node.
#     :min_samples_split => 2,
#     # The minimum number of samples required to be at a leaf node.
#     :min_samples_leaf => 1,
#     # The fraction of samples to be used for fitting the individual base
#     # learners.
#     :subsample => 1.0,
#     # Number of features to consider when looking for the best split.
#     # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
#     :max_features => "auto",
#     # An estimator object that is used to compute the initial predictions.
#     # (Python BaseEstimator, nothing)
#     :init => nothing,
#     # Enable verbose output.
#     # (0, 1, >1)
#     :verbose => 0
#   },
# }
# </pre>
type SKLGradientBoosting <: SKLLearner
  model
  options
  
  function SKLGradientBoosting(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Loss function to be optimized. 
        # ("deviance")
        :loss => "deviance",
        # Learning rate shrinks the contribution of each tree by learning_rate.
        :learning_rate => 0.1,
        # The number of boosting stages to perform.
        :n_estimators => 100,
        # Maximum depth of the individual regression estimators. 
        :max_depth => 3,
        # The minimum number of samples required to split an internal node.
        :min_samples_split => 2,
        # The minimum number of samples required to be at a leaf node.
        :min_samples_leaf => 1,
        # The fraction of samples to be used for fitting the individual base
        # learners.
        :subsample => 1.0,
        # Number of features to consider when looking for the best split.
        # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
        :max_features => "auto",
        # An estimator object that is used to compute the initial predictions.
        # (Python BaseEstimator, nothing)
        :init => nothing,
        # Enable verbose output.
        # (0, 1, >1)
        :verbose => 0
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLGradientBoosting ENS.GradientBoostingClassifier


# Logistic Regression.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Used to specify the norm used in the penalization.
#     # ("l1", "l2")
#     :penalty => "l2",
#     # Dual or primal formulation. 
#     # Dual formulation is only implemented for l2 penalty.
#     :dual => false,
#     # Tolerance for stopping criteria.
#     :tol => 0.0001,
#     # Inverse of regularization strength, must be positive float.
#     :C => 1.0,
#     # Specifies if a constant (a.k.a. bias or intercept) should be added the
#     # decision function.
#     :fit_intercept => true,
#     # TODO(svs14): Simplify explanation sci-kit learn provides.
#     :intercept_scaling => 1,
#     # TODO(svs14): Simplify explanation sci-kit learn provides.
#     :class_weight => nothing,
#     # Undocumented in sci-kit learn.
#     :random_state => nothing
#   },
# }
# </pre>
type SKLLogisticRegression <: SKLLearner
  model
  options
  
  function SKLLogisticRegression(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Used to specify the norm used in the penalization.
        # ("l1", "l2")
        :penalty => "l2",
        # Dual or primal formulation. 
        # Dual formulation is only implemented for l2 penalty.
        :dual => false,
        # Tolerance for stopping criteria.
        :tol => 0.0001,
        # Inverse of regularization strength, must be positive float.
        :C => 1.0,
        # Specifies if a constant (a.k.a. bias or intercept) should be added the
        # decision function.
        :fit_intercept => true,
        # TODO(svs14): Simplify explanation sci-kit learn provides.
        :intercept_scaling => 1,
        # TODO(svs14): Simplify explanation sci-kit learn provides.
        :class_weight => nothing,
        # Undocumented in sci-kit learn.
        :random_state => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLLogisticRegression LM.LogisticRegression


# Passive Aggressive.
#
# <pre> 
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Maximum step size (regularization).
#     :C => 1.0,
#     # Whether the intercept should be estimated or not.
#     # If false, the data is assumed to be already centered.
#     :fit_intercept => true,
#     # The number of passes over the training data (aka epochs).
#     :n_iter => 5,
#     # Whether or not the training data should be shuffled after each epoch.
#     :shuffle => false,
#     # Verbosity level.
#     :verbose => 0,
#     # The loss function to be used.
#     # ("hinge", "squared_hinge")
#     :loss => "hinge",
#     # Number of CPUs to use in multi-class problems.
#     :n_jobs => 1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing,
#     # When set to True, reuse the solution of the previous call to fit as
#     # initialization, otherwise, just erase the previous solution.
#     :warm_start => false
#   },
# }
# </pre> 
type SKLPassiveAggressive <: SKLLearner
  model
  options
  
  function SKLPassiveAggressive(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Maximum step size (regularization).
        :C => 1.0,
        # Whether the intercept should be estimated or not.
        # If false, the data is assumed to be already centered.
        :fit_intercept => true,
        # The number of passes over the training data (aka epochs).
        :n_iter => 5,
        # Whether or not the training data should be shuffled after each epoch.
        :shuffle => false,
        # Verbosity level.
        :verbose => 0,
        # The loss function to be used.
        # ("hinge", "squared_hinge")
        :loss => "hinge",
        # Number of CPUs to use in multi-class problems.
        :n_jobs => 1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # When set to True, reuse the solution of the previous call to fit as
        # initialization, otherwise, just erase the previous solution.
        :warm_start => false
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLPassiveAggressive LM.PassiveAggressiveClassifier


# Linear least squares with l2 regulariation.
# 
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Small positive values of alpha improve the conditioning of the problem
#     # and reduce the variance of the estimates.
#     :alpha => 1.0,
#     # Whether to calculate the intercept for this model.
#     :fit_intercept => true,
#     # If True, the regressors X will be normalized before regression.
#     :normalize => false,
#     # If True, X will be copied; else, it may be overwritten.
#     :copy_X => true,
#     # Maximum number of iterations for conjugate gradient solver.
#     :max_iter => nothing,
#     # Precision of the solution.
#     :tol => 0.001,
#     # Weights associated with classes in the form {class_label : weight}. If
#     # not given, all classes are supposed to have weight one.
#     :class_weight => nothing,
#     # TODO(svs14): Summarize this.
#     # Solver to use in the computational routines. ‘svd’ will use a Singular
#     # value decomposition to obtain the solution, ‘dense_cholesky’ will use
#     # the standard scipy.linalg.solve function, ‘sparse_cg’ will use the
#     # conjugate gradient solver as found in scipy.sparse.linalg.cg while
#     # ‘auto’ will chose the most appropriate depending on the matrix X.
#     # ‘lsqr’ uses a direct regularized least-squares routine provided by
#     # scipy.
#     # ("auto", "svd", "dense_cholesky", "lsqr", "sparse_cg")
#     :solver => "auto"
#   },
# }
# </pre>
type SKLRidge <: SKLLearner
  model
  options
  
  function SKLRidge(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Small positive values of alpha improve the conditioning of the problem
        # and reduce the variance of the estimates.
        :alpha => 1.0,
        # Whether to calculate the intercept for this model.
        :fit_intercept => true,
        # If True, the regressors X will be normalized before regression.
        :normalize => false,
        # If True, X will be copied; else, it may be overwritten.
        :copy_X => true,
        # Maximum number of iterations for conjugate gradient solver.
        :max_iter => nothing,
        # Precision of the solution.
        :tol => 0.001,
        # Weights associated with classes in the form {class_label : weight}. If
        # not given, all classes are supposed to have weight one.
        :class_weight => nothing,
        # TODO(svs14): Summarize this.
        # Solver to use in the computational routines. ‘svd’ will use a Singular
        # value decomposition to obtain the solution, ‘dense_cholesky’ will use
        # the standard scipy.linalg.solve function, ‘sparse_cg’ will use the
        # conjugate gradient solver as found in scipy.sparse.linalg.cg while
        # ‘auto’ will chose the most appropriate depending on the matrix X.
        # ‘lsqr’ uses a direct regularized least-squares routine provided by
        # scipy.
        # ("auto", "svd", "dense_cholesky", "lsqr", "sparse_cg")
        :solver => "auto"
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLRidge LM.RidgeClassifier

# Ridge with built-in cross-validation.
#
# <pre>
# default_options = {
#   # Metrcvc to train against
#   # (:accuracy).
#   :metrcvc => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Array of alpha values to try. Small positive values of alpha improve
#     # the conditioning of the problem and reduce the variance of the
#     # estimates.
#     :alphas => [0.1, 1., 10.],
#     # Whether to calculate the intercept for this model.
#     :fit_intercept => true,
#     # If True, the regressors X will be normalized before regression.
#     :normalize => false,
#     # Very Python specific (see original sci-kit learn documentation).
#     :score_func => nothing,
#     # Very Python specific (see original sci-kit learn documentation).
#     :loss_func => nothing,
#     # Very Python specific (see original sci-kit learn documentation).
#     :cv => nothing,
#     # Weights associated with classes in the form {class_label : weight}.
#     :class_weight => nothing
#   },
# }
# </pre>
type SKLRidgeCV <: SKLLearner
  model
  options
  
  function SKLRidgeCV(options=Dict())
    default_options = {
      # Metrcvc to train against
      # (:accuracy).
      :metrcvc => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Array of alpha values to try. Small positive values of alpha improve
        # the conditioning of the problem and reduce the variance of the
        # estimates.
        :alphas => [0.1, 1., 10.],
        # Whether to calculate the intercept for this model.
        :fit_intercept => true,
        # If True, the regressors X will be normalized before regression.
        :normalize => false,
        # Very Python specific (see original sci-kit learn documentation).
        :score_func => nothing,
        # Very Python specific (see original sci-kit learn documentation).
        :loss_func => nothing,
        # Very Python specific (see original sci-kit learn documentation).
        :cv => nothing,
        # Weights associated with classes in the form {class_label : weight}.
        :class_weight => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLRidgeCV LM.RidgeClassifierCV

# Linear classifiers with SGD training.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # The loss function to be used. Defaults to ‘hinge’, which gives a
#     # linear SVM. The ‘log’ loss gives logistic regression, a probabilistic
#     # classifier. ‘modified_huber’ is another smooth loss that brings
#     # tolerance to outliers as well as probability estimates.
#     # ‘squared_hinge’ is like hinge but is quadratically penalized.
#     # ‘perceptron’ is the linear loss used by the perceptron algorithm. The
#     # other losses are designed for regression but can be useful in
#     # classification as well.
#     # ("hinge", "log", "modified_huber", "squared_hinge",
#     #  "perceptron", "squared_loss", "huber", "epsilon_insensitive",
#     #  "squared_epsilon_insensitive")
#     :loss => "hinge",
#     # The penalty (aka regularization term) to be used.
#     # ("l2", "l1", "elasticnet")
#     :penalty => "l2",
#     # Constant that multiplies the regularization term. 
#     :alpha => 0.0001,
#     # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0
#     # corresponds to L2 penalty, l1_ratio=1 to L1.
#     :l1_ratio => 0.15,
#     # Whether the intercept should be estimated or not.
#     :fit_intercept => true,
#     # The number of passes over the training data (aka epochs).
#     :n_iter => 5,
#     # Whether or not the training data should be shuffled after each epoch.
#     :shuffle => false,
#     # The verbosity level.
#     :verbose => 0,
#     # Epsilon in the epsilon-insensitive loss functions; only if loss is
#     # ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For
#     # ‘huber’, determines the threshold at which it becomes less important
#     # to get the prediction exactly right. For epsilon-insensitive, any
#     # differences between the current prediction and the correct label are
#     # ignored if they are less than this threshold.
#     :epsilon => 0.1,
#     # The number of CPUs to use for multi-class problems.
#     :n_jobs => 1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing,
#     # The learning rate: constant: eta = eta0 optimal: eta = 1.0/(t+t0)
#     # [default] invscaling: eta = eta0 / pow(t, power_t) .
#     :learning_rate => "optimal",
#     # The initial learning rate.
#     :eta0 => 0.0,
#     # The exponent for inverse scaling learning rate.
#     :power_t => 0.5,
#     # Very Python specific (see original sci-kit learn documentation).
#     :class_weight => nothing,
#     # When set to True, reuse the solution of the previous call to fit as
#     # initialization, otherwise, just erase the previous solution.
#     :warm_start => false,
#     # Undocumented.
#     :rho => nothing,
#     # Undocumented.
#     :seed => nothing
#   },
# }
# </pre>
type SKLSGD <: SKLLearner
  model
  options
  
  function SKLSGD(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # The loss function to be used. Defaults to ‘hinge’, which gives a
        # linear SVM. The ‘log’ loss gives logistic regression, a probabilistic
        # classifier. ‘modified_huber’ is another smooth loss that brings
        # tolerance to outliers as well as probability estimates.
        # ‘squared_hinge’ is like hinge but is quadratically penalized.
        # ‘perceptron’ is the linear loss used by the perceptron algorithm. The
        # other losses are designed for regression but can be useful in
        # classification as well.
        # ("hinge", "log", "modified_huber", "squared_hinge",
        #  "perceptron", "squared_loss", "huber", "epsilon_insensitive",
        #  "squared_epsilon_insensitive")
        :loss => "hinge",
        # The penalty (aka regularization term) to be used.
        # ("l2", "l1", "elasticnet")
        :penalty => "l2",
        # Constant that multiplies the regularization term. 
        :alpha => 0.0001,
        # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0
        # corresponds to L2 penalty, l1_ratio=1 to L1.
        :l1_ratio => 0.15,
        # Whether the intercept should be estimated or not.
        :fit_intercept => true,
        # The number of passes over the training data (aka epochs).
        :n_iter => 5,
        # Whether or not the training data should be shuffled after each epoch.
        :shuffle => false,
        # The verbosity level.
        :verbose => 0,
        # Epsilon in the epsilon-insensitive loss functions; only if loss is
        # ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For
        # ‘huber’, determines the threshold at which it becomes less important
        # to get the prediction exactly right. For epsilon-insensitive, any
        # differences between the current prediction and the correct label are
        # ignored if they are less than this threshold.
        :epsilon => 0.1,
        # The number of CPUs to use for multi-class problems.
        :n_jobs => 1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # The learning rate: constant: eta = eta0 optimal: eta = 1.0/(t+t0)
        # [default] invscaling: eta = eta0 / pow(t, power_t) .
        :learning_rate => "optimal",
        # The initial learning rate.
        :eta0 => 0.0,
        # The exponent for inverse scaling learning rate.
        :power_t => 0.5,
        # Very Python specific (see original sci-kit learn documentation).
        :class_weight => nothing,
        # When set to True, reuse the solution of the previous call to fit as
        # initialization, otherwise, just erase the previous solution.
        :warm_start => false,
        # Undocumented.
        :rho => nothing,
        # Undocumented.
        :seed => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLSGD LM.SGDClassifier

# K nearest neighbors.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Number of neighbors to use.
#     :n_neighbors => 5,
#     # Weight function used in prediction.
#     # ("uniform", "distance", Function - see scikit-learn documentation)
#     :weights => "uniform",
#     # Algorithm used to compute the nearest neighbors.
#     # ("ball_tree", "kd_tree", "brute", "auto")
#     :algorithm => "auto",
#     # Leaf size passed to BallTree or KDTree.
#     :leaf_size => 30,
#     # Power parameter for the Minkowski metric. When p = 1, this is
#     # equivalent to using manhattan_distance (l1), and euclidean_distance
#     # (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
#     :p => 2,
#     # The distance metric to use for the tree. The default metric is
#     # minkowski, and with p=2 is equivalent to the standard Euclidean
#     # metric.
#     :metric => "minkowski"
#   },
# }
# </pre>
type SKLKNeighbors <: SKLLearner
  model
  options
  
  function SKLKNeighbors(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Number of neighbors to use.
        :n_neighbors => 5,
        # Weight function used in prediction.
        # ("uniform", "distance", Function - see scikit-learn documentation)
        :weights => "uniform",
        # Algorithm used to compute the nearest neighbors.
        # ("ball_tree", "kd_tree", "brute", "auto")
        :algorithm => "auto",
        # Leaf size passed to BallTree or KDTree.
        :leaf_size => 30,
        # Power parameter for the Minkowski metric. When p = 1, this is
        # equivalent to using manhattan_distance (l1), and euclidean_distance
        # (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        :p => 2,
        # The distance metric to use for the tree. The default metric is
        # minkowski, and with p=2 is equivalent to the standard Euclidean
        # metric.
        :metric => "minkowski"
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLKNeighbors NN.KNeighborsClassifier

# K nearest neighbors within radius.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Range of parameter space to use by default for :meth`radius_neighbors`
#     # queries.
#     :radius => 1.0,
#     # Weight function used in prediction.
#     # ("uniform", "distance", Function - see scikit-learn documentation)
#     :weights => "uniform",
#     # Algorithm used to compute the nearest neighbors.
#     # ("ball_tree", "kd_tree", "brute", "auto")
#     :algorithm => "auto",
#     # Leaf size passed to BallTree or KDTree.
#     :leaf_size => 30,
#     # Power parameter for the Minkowski metric. When p = 1, this is
#     # equivalent to using manhattan_distance (l1), and euclidean_distance
#     # (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
#     :p => 2,
#     # The distance metric to use for the tree. The default metric is
#     # minkowski, and with p=2 is equivalent to the standard Euclidean
#     # metric.
#     :metric => "minkowski",
#     # NOTE(svs14): Unlike sci-kit learn, we override nothing option 
#     #              with a random label for outliers.
#     # Label, which is given for outlier samples (samples with no neighbors
#     # on given radius). If set to None, ValueError is raised, when outlier
#     # is detected.
#     :outlier_label => nothing
#   },
# }
# </pre>
type SKLRadiusNeighbors <: SKLLearner
  model
  options
  
  function SKLRadiusNeighbors(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Range of parameter space to use by default for :meth`radius_neighbors`
        # queries.
        :radius => 1.0,
        # Weight function used in prediction.
        # ("uniform", "distance", Function - see scikit-learn documentation)
        :weights => "uniform",
        # Algorithm used to compute the nearest neighbors.
        # ("ball_tree", "kd_tree", "brute", "auto")
        :algorithm => "auto",
        # Leaf size passed to BallTree or KDTree.
        :leaf_size => 30,
        # Power parameter for the Minkowski metric. When p = 1, this is
        # equivalent to using manhattan_distance (l1), and euclidean_distance
        # (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        :p => 2,
        # The distance metric to use for the tree. The default metric is
        # minkowski, and with p=2 is equivalent to the standard Euclidean
        # metric.
        :metric => "minkowski",
        # NOTE(svs14): Unlike sci-kit learn, we override nothing option 
        #              with a random label for outliers.
        # Label, which is given for outlier samples (samples with no neighbors
        # on given radius). If set to None, ValueError is raised, when outlier
        # is detected.
        :outlier_label => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

function train!(rn::SKLRadiusNeighbors, instances::Matrix, labels::Vector)
  impl_options = copy(rn.options[:impl_options])

  # Set training-dependent options
  if impl_options[:outlier_label] == nothing
    # Randomly select a label to assign to outliers
    impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
  end

  # Train model
  rn.model = NN.RadiusNeighborsClassifier(;impl_options...)
  rn.model[:fit](instances, labels)
end

# Nearest centroid.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Very Python specific (see original sci-kit learn documentation).
#     :metric => "euclidean",
#     # Threshold for shrinking centroids to remove features.
#     # (Float, nothing)
#     :shrink_threshold => nothing
#   },
# }
# </pre>
type SKLNearestCentroid <: SKLLearner
  model
  options
  
  function SKLNearestCentroid(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Very Python specific (see original sci-kit learn documentation).
        :metric => "euclidean",
        # Threshold for shrinking centroids to remove features.
        # (Float, nothing)
        :shrink_threshold => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLNearestCentroid NN.NearestCentroid

# C-Support Vector Classification.
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Penalty parameter C of the error term.
#     :C => 1.0,
#     # Specifies the kernel type to be used in the algorithm.
#     # ("linear", "poly", "rbf", "sigmoid", "precomputed", Python callable)
#     :kernel => "rbf",
#     # Degree of the polynomial kernel function (‘poly’). Ignored by all
#     # other kernels.
#     :degree => 3,
#     # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigm’. If gamma is 0.0 then
#     # 1/n_features will be used instead.
#     :gamma => 0.0,
#     # Independent term in kernel function. It is only significant in ‘poly’
#     # and ‘sigmoid’.
#     :coef0 => 0.0,
#     # Whether to use the shrinking heuristic.
#     :shrinking => true,
#     # Whether to enable probability estimates. This must be enabled prior to
#     # calling fit, and will slow down that method.
#     :probability => false,
#     # Tolerance for stopping criterion.
#     :tol => 0.001,
#     # Specify the size of the kernel cache (in MB)
#     :cache_size => 200,
#     # Very Python specific (see original sci-kit learn documentation).
#     :class_weight => nothing,
#     # Enable verbose output.
#     :verbose => false,
#     # Hard limit on iterations within solver, or -1 for no limit.
#     :max_iter => -1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing
#   },
# }
# </pre>
type SKLSVC <: SKLLearner
  model
  options
  
  function SKLSVC(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Penalty parameter C of the error term.
        :C => 1.0,
        # Specifies the kernel type to be used in the algorithm.
        # ("linear", "poly", "rbf", "sigmoid", "precomputed", Python callable)
        :kernel => "rbf",
        # Degree of the polynomial kernel function (‘poly’). Ignored by all
        # other kernels.
        :degree => 3,
        # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigm’. If gamma is 0.0 then
        # 1/n_features will be used instead.
        :gamma => 0.0,
        # Independent term in kernel function. It is only significant in ‘poly’
        # and ‘sigmoid’.
        :coef0 => 0.0,
        # Whether to use the shrinking heuristic.
        :shrinking => true,
        # Whether to enable probability estimates. This must be enabled prior to
        # calling fit, and will slow down that method.
        :probability => false,
        # Tolerance for stopping criterion.
        :tol => 0.001,
        # Specify the size of the kernel cache (in MB)
        :cache_size => 200,
        # Very Python specific (see original sci-kit learn documentation).
        :class_weight => nothing,
        # Enable verbose output.
        :verbose => false,
        # Hard limit on iterations within solver, or -1 for no limit.
        :max_iter => -1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLSVC SVM.SVC

# Linear Support Vector Classifier.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # Specifies the norm used in the penalization. 
#     # ("l1", "l2")
#     :penalty => "l2",
#     # Specifies the loss function.
#     # ("l1", "l2")
#     :loss => "l2",
#     # Select the algorithm to either solve the dual or primal optimization
#     # problem.
#     :dual => true,
#     # Tolerance for stopping criteria.
#     :tol => 0.0001,
#     # Penalty parameter C of the error term.
#     :C => 1.0,
#     # Determines the multi-class strategy if y contains more than two
#     # classes.
#     # ("ovr", "crammer_singer")
#     :multi_class => "ovr",
#     # Whether to calculate the intercept for this model. 
#     :fit_intercept => true,
#     # TODO(svs14): Simplify explanation sci-kit learn provides.
#     :intercept_scaling => 1,
#     # TODO(svs14): Simplify explanation sci-kit learn provides.
#     :class_weight => nothing,
#     # Enable verbose output.
#     :verbose => 0,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing
#   },
# }
# </pre>
type SKLLinearSVC <: SKLLearner
  model
  options
  
  function SKLLinearSVC(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # Specifies the norm used in the penalization. 
        # ("l1", "l2")
        :penalty => "l2",
        # Specifies the loss function.
        # ("l1", "l2")
        :loss => "l2",
        # Select the algorithm to either solve the dual or primal optimization
        # problem.
        :dual => true,
        # Tolerance for stopping criteria.
        :tol => 0.0001,
        # Penalty parameter C of the error term.
        :C => 1.0,
        # Determines the multi-class strategy if y contains more than two
        # classes.
        # ("ovr", "crammer_singer")
        :multi_class => "ovr",
        # Whether to calculate the intercept for this model. 
        :fit_intercept => true,
        # TODO(svs14): Simplify explanation sci-kit learn provides.
        :intercept_scaling => 1,
        # TODO(svs14): Simplify explanation sci-kit learn provides.
        :class_weight => nothing,
        # Enable verbose output.
        :verbose => 0,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLLinearSVC SVM.LinearSVC

# Nu-Support Vector Classifier.
#
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # An upper bound on the fraction of training errors and a lower bound of
#     # the fraction of support vectors. Should be in the interval (0, 1].
#     :nu => 0.5,
#     # Specifies the kernel type to be used in the algorithm.
#     # ("linear", "poly", "rbf", "sigmoid", "precomputed", Python callable)
#     :kernel => "rbf",
#     # Degree of kernel function is significant only in poly, rbf, sigmoid.
#     :degree => 3,
#     # Kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features
#     # will be taken.
#     :gamma => 0.0,
#     # Independent term in kernel function. It is only significant in
#     # poly/sigmoid.
#     :coef0 => 0.0,
#     # Whether to use the shrinking heuristic.
#     :shrinking => true,
#     # Whether to enable probability estimates. This must be enabled prior to
#     # calling fit, and will slow down that method.
#     :probability => false,
#     # Tolerance for stopping criterion.
#     :tol => 0.001,
#     # Specify the size of the kernel cache (in MB).
#     :cache_size => 200,
#     # Enable verbose output.
#     :verbose => false,
#     # Hard limit on iterations within solver, or -1 for no limit.
#     :max_iter => -1,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing
#   },
# }
# </pre>
type SKLNuSVC <: SKLLearner
  model
  options
  
  function SKLNuSVC(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # An upper bound on the fraction of training errors and a lower bound of
        # the fraction of support vectors. Should be in the interval (0, 1].
        :nu => 0.5,
        # Specifies the kernel type to be used in the algorithm.
        # ("linear", "poly", "rbf", "sigmoid", "precomputed", Python callable)
        :kernel => "rbf",
        # Degree of kernel function is significant only in poly, rbf, sigmoid.
        :degree => 3,
        # Kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features
        # will be taken.
        :gamma => 0.0,
        # Independent term in kernel function. It is only significant in
        # poly/sigmoid.
        :coef0 => 0.0,
        # Whether to use the shrinking heuristic.
        :shrinking => true,
        # Whether to enable probability estimates. This must be enabled prior to
        # calling fit, and will slow down that method.
        :probability => false,
        # Tolerance for stopping criterion.
        :tol => 0.001,
        # Specify the size of the kernel cache (in MB).
        :cache_size => 200,
        # Enable verbose output.
        :verbose => false,
        # Hard limit on iterations within solver, or -1 for no limit.
        :max_iter => -1,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLNuSVC SVM.NuSVC


# Decision tree classifier.
# 
# <pre>
# default_options = {
#   # Metric to train against
#   # (:accuracy).
#   :metric => :accuracy,
#   # Options specific to this implementation.
#   :impl_options => {
#     # The function to measure the quality of a split.
#     # ("gini", "entropy")
#     :criterion => "gini",
#     # Undocumented.
#     :splitter => "best",
#     # The maximum depth of the tree. If None, then nodes are expanded until
#     # all leaves are pure or until all leaves contain less than
#     # min_samples_split samples.
#     :max_depth => nothing,
#     # The minimum number of samples required to split an internal node.
#     :min_samples_split => 2,
#     # The minimum number of samples required to be at a leaf node.
#     :min_samples_leaf => 1,
#     # Number of features to consider when looking for the best split.
#     # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
#     :max_features => nothing,
#     # The seed of the pseudo random number generator to use when shuffling
#     # the data.
#     # (Int, Python RandomState, nothing)
#     :random_state => nothing,
#     # Undocumented.
#     :min_density => nothing,
#     # Undocumented.
#     :compute_importances => nothing
#   },
# }
# </pre>
type SKLDecisionTree <: SKLLearner
  model
  options
  
  function SKLDecisionTree(options=Dict())
    default_options = {
      # Metric to train against
      # (:accuracy).
      :metric => :accuracy,
      # Options specific to this implementation.
      :impl_options => {
        # The function to measure the quality of a split.
        # ("gini", "entropy")
        :criterion => "gini",
        # Undocumented.
        :splitter => "best",
        # The maximum depth of the tree. If None, then nodes are expanded until
        # all leaves are pure or until all leaves contain less than
        # min_samples_split samples.
        :max_depth => nothing,
        # The minimum number of samples required to split an internal node.
        :min_samples_split => 2,
        # The minimum number of samples required to be at a leaf node.
        :min_samples_leaf => 1,
        # Number of features to consider when looking for the best split.
        # (Int, Float - acts as percentage, "auto", "sqrt", "log2", nothing)
        :max_features => nothing,
        # The seed of the pseudo random number generator to use when shuffling
        # the data.
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # Undocumented.
        :min_density => nothing,
        # Undocumented.
        :compute_importances => nothing
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end

@build_train!_func SKLDecisionTree TREE.DecisionTreeClassifier


end # module
