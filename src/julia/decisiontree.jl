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

function fit!(tree::PrunedTree,
  instances::Matrix{Float64}, labels::Vector{Float64})

  impl_options = tree.options[:impl_options]

  tree.model[:impl] = Dict()
  tree.model[:impl][:output] = tree.options[:output]
  # Convert labels if classification problem
  if tree.options[:output] == :class
    labels = convert(Vector{Int}, labels)
  end

  tree.model[:impl][:tree] = DT.build_tree(labels, instances)
  tree.model[:impl][:tree] = DT.prune_tree(
    tree.model[:impl][:tree], impl_options[:purity_threshold]
  )

  return tree
end

function transform!(tree::PrunedTree,
  instances::Matrix{Float64})

  # Obtain predictions
  predictions = DT.apply_tree(tree.model[:impl][:tree], instances)

  # Convert labels if classification problem
  if tree.model[:impl][:output] == :class
    predictions = convert(Vector{Float64}, predictions)
  end

  return predictions
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

function fit!(forest::RandomForest,
  instances::Matrix{Float64}, labels::Vector{Float64})

  # Set training-dependent options
  impl_options = forest.options[:impl_options]
  if impl_options[:num_subfeatures] == nothing
    num_subfeatures = size(instances, 2)
  else
    num_subfeatures = impl_options[:num_subfeatures]
  end

  forest.model[:impl] = Dict()
  forest.model[:impl][:output] = forest.options[:output]
  # Convert labels if classification problem
  if forest.options[:output] == :class
    labels = convert(Vector{Int}, labels)
  end

  # Build model
  forest.model[:impl][:forest] = DT.build_forest(
    labels, 
    instances,
    num_subfeatures, 
    impl_options[:num_trees],
    impl_options[:partial_sampling]
  )

  return forest
end

function transform!(forest::RandomForest,
  instances::Matrix{Float64})

  # Obtain predictions
  predictions = DT.apply_forest(forest.model[:impl][:forest], instances)

  # Convert labels if classification problem
  if forest.model[:impl][:output] == :class
    predictions = convert(Vector{Float64}, predictions)
  end

  return predictions
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
  instances::Matrix{Float64}, labels::Vector{Float64})

  adaboost.model[:impl] = Dict()
  adaboost.model[:impl][:output] = adaboost.options[:output]
  # Convert labels if classification problem
  if adaboost.options[:output] == :class
    labels = convert(Vector{Int}, labels)
  end

  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within Orchestra.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, instances, adaboost.options[:impl_options][:num_iterations]
  )

  adaboost.model[:impl][:adaboost] = {
    :ensemble => ensemble,
    :coefficients => coefficients
  }

  return adaboost
end

function transform!(adaboost::DecisionStumpAdaboost,
  instances::Matrix{Float64})

  # Obtain predictions
  predictions = DT.apply_adaboost_stumps(
    adaboost.model[:impl][:adaboost][:ensemble],
    adaboost.model[:impl][:adaboost][:coefficients],
    instances
  )

  # Convert labels if classification problem
  if adaboost.model[:impl][:output] == :class
    predictions = convert(Vector{Float64}, predictions)
  end

  return predictions
end

end # module
