# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

importall Orchestra.AbstractLearner
using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM

export SKLRandomForest,
       SKLExtraTrees,
       SKLGradientBoosting,
       SKLLogisticRegression,
       train!,
       predict!

type SKLRandomForest <: Learner
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
        # If Int, random_state is the seed used by the random number generator. 
        # If Python RandomState instance, 
        # random_state is the random number generator.
        # If nothing, the random number generator 
        # is the Python RandomState instance used by Python np.random. 
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # Controls the verbosity of the tree building process.
        :verbose => 0
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end


function train!(rf::SKLRandomForest, instances::Matrix, labels::Vector)
  impl_options = rf.options[:impl_options]
  rf.model = ENS.RandomForestClassifier(;impl_options...)
  rf.model[:fit](instances, labels)
end

function predict!(rf::SKLRandomForest, instances::Matrix)
  return collect(rf.model[:predict](instances))
end

type SKLExtraTrees <: Learner
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
        # If Int, random_state is the seed used by the random number generator. 
        # If Python RandomState instance, 
        # random_state is the random number generator.
        # If nothing, the random number generator 
        # is the Python RandomState instance used by Python np.random. 
        # (Int, Python RandomState, nothing)
        :random_state => nothing,
        # Controls the verbosity of the tree building process.
        :verbose => 0
      },
    }
    new(nothing, merge(default_options, options)) 
  end
end


function train!(et::SKLExtraTrees, instances::Matrix, labels::Vector)
  impl_options = et.options[:impl_options]
  et.model = ENS.ExtraTreesClassifier(;impl_options...)
  et.model[:fit](instances, labels)
end

function predict!(et::SKLExtraTrees, instances::Matrix)
  return collect(et.model[:predict](instances))
end

type SKLGradientBoosting <: Learner
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


function train!(gb::SKLGradientBoosting, instances::Matrix, labels::Vector)
  impl_options = gb.options[:impl_options]
  gb.model = ENS.GradientBoostingClassifier(;impl_options...)
  gb.model[:fit](instances, labels)
end

function predict!(gb::SKLGradientBoosting, instances::Matrix)
  return collect(gb.model[:predict](instances))
end

type SKLLogisticRegression <: Learner
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


function train!(lr::SKLLogisticRegression, instances::Matrix, labels::Vector)
  impl_options = lr.options[:impl_options]
  lr.model = LM.LogisticRegression(;impl_options...)
  lr.model[:fit](instances, labels)
end

function predict!(lr::SKLLogisticRegression, instances::Matrix)
  return collect(lr.model[:predict](instances))
end

end # module
