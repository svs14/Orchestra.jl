# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

importall Orchestra.AbstractLearner
using PyCall
@pyimport sklearn.ensemble as ENS

export SKLRandomForest,
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

end # module
