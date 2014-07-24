# Decision trees as found in DecisionTree Julia package.
module DecisionTreeWrapper

importall Orchestra.Types
importall Orchestra.Util

import DecisionTree
DT = DecisionTree

export PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       fit!, 
       transform!

# Pruned ID3 decision tree.
type PrunedTree <: Learner
  model::Dict
  options::Dict
  
  function PrunedTree(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Merge leaves having >= purity_threshold combined purity.
        :purity_threshold => 1.0
      }
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(tree::PrunedTree, instances::Matrix, labels::Vector)
  impl_options = tree.options[:impl_options]

  tree.model[:impl] = DT.build_tree(labels, instances)
  tree.model[:impl] = DT.prune_tree(
    tree.model[:impl], impl_options[:purity_threshold]
  )

  return tree
end
function transform!(tree::PrunedTree, instances::Matrix)
  return DT.apply_tree(tree.model[:impl], instances)
end

# Random forest (C4.5).
type RandomForest <: Learner
  model::Dict
  options::Dict
  
  function RandomForest(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Number of features to train on with trees.
        :num_subfeatures => nothing,
        # Number of trees in forest.
        :num_trees => 10,
        # Proportion of trainingset to be used for trees.
        :partial_sampling => 0.7
      }
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(forest::RandomForest, instances::Matrix, labels::Vector)
  # Set training-dependent options
  impl_options = forest.options[:impl_options]
  if impl_options[:num_subfeatures] == nothing
    num_subfeatures = size(instances, 2)
  else
    num_subfeatures = impl_options[:num_subfeatures]
  end

  # Build model
  forest.model[:impl] = DT.build_forest(
    labels, 
    instances,
    num_subfeatures, 
    impl_options[:num_trees],
    impl_options[:partial_sampling]
  )

  return forest
end

function transform!(forest::RandomForest, instances::Matrix)
  return DT.apply_forest(forest.model[:impl], instances)
end

# Adaboosted C4.5 decision stumps.
type DecisionStumpAdaboost <: Learner
  model::Dict
  options::Dict
  
  function DecisionStumpAdaboost(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Number of boosting iterations.
        :num_iterations => 7
      }
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(adaboost::DecisionStumpAdaboost, 
  instances::Matrix, labels::Vector)

  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within Orchestra.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, instances, adaboost.options[:impl_options][:num_iterations]
  )

  adaboost.model[:impl] = {
    :ensemble => ensemble,
    :coefficients => coefficients
  }

  return adaboost
end

function transform!(adaboost::DecisionStumpAdaboost, instances::Matrix)
  return DT.apply_adaboost_stumps(
    adaboost.model[:impl][:ensemble],
    adaboost.model[:impl][:coefficients],
    instances
  )
end

end # module
