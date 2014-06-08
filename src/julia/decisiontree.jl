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

# Converts labels into strings if numeric labels used
# in classification, otherwise returns as is.
function label_convert(target_output, labels)
  label_conversion_required = false
  label_type = eltype(labels)
  if target_output == :class && label_type <: Real
    label_conversion_required = true
    labels = map(x -> string(x), labels)
  end
  if target_output == :regression && !issubtype(label_type, FloatingPoint)
    label_conversion_required = true
    labels = convert(Vector{Float64}, labels)
  end
  return (label_conversion_required, labels)
end

# Conversion of instances occurs when Matrix{Any} is really Matrix{Real}
# when under regression.
# Fails if output is regression and matrix is not Real typed.
function instance_convert(target_output, instances)
  instance_conversion_required = false
  if !issubtype(eltype(instances), Real) && target_output == :regression
    instances = promote(instances)[1]
    if eltype(instances) <: Real
      instance_conversion_required = true
    else
      error("DecisionTree only supports numeric features for regression")
    end
  end
  return instance_conversion_required, instances
end

# Pruned ID3 decision tree.
#
# Only support numeric features when using regression.
type PrunedTree <: Learner
  model
  options
  
  function PrunedTree(options=Dict())
    default_options = {
      # Target output.
      # (:class, :regression).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Merge leaves having >= purity_threshold combined purity.
        :purity_threshold => 1.0
      }
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(tree::PrunedTree, instances::Matrix, labels::Vector)
  # Workaround of DT API when numeric labels are used in classification, and
  # regression with numeric labels (Real, not FloatingPoint)
  label_conversion_required, labels = 
    label_convert(tree.options[:output], labels)

  # Conversion when instance is Matrix{Any} but is really Matrix{Real}
  instance_conversion_required, instances = 
    instance_convert(tree.options[:output], instances)

  tree.model = Dict()
  tree.model[:output] = tree.options[:output]
  tree.model[:instance_conversion_required] = instance_conversion_required
  tree.model[:label_conversion_required] = label_conversion_required

  # Train
  impl_options = tree.options[:impl_options]
  tree.model[:model] = DT.build_tree(labels, instances)
  tree.model[:model] = 
    DT.prune_tree(tree.model[:model], impl_options[:purity_threshold])

  tree.model
end

function transform!(tree::PrunedTree, instances::Matrix)
  if tree.model[:output] == :regression &&
    tree.model[:instance_conversion_required]

    instances = promote(instances)[1]
  end
  predictions = DT.apply_tree(tree.model[:model], instances)

  if tree.model[:output] == :class && 
    tree.model[:label_conversion_required]
    
    predictions = map(x -> float(x), predictions)
    predictions = [promote(predictions...)...]
  end

  return predictions
end

# Random forest (C4.5).
#
# Only support numeric features when using regression.
type RandomForest <: Learner
  model
  options
  
  function RandomForest(options=Dict())
    default_options = {
      # Target output.
      # (:class, :regression).
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
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(forest::RandomForest, instances::Matrix, labels::Vector)
  # Workaround of DT API when numeric labels are used in classification, and
  # regression with numeric labels (Real, not FloatingPoint)
  label_conversion_required, labels = 
    label_convert(forest.options[:output], labels)

  # Conversion when instance is Matrix{Any} but is really Matrix{Real}
  instance_conversion_required, instances = 
    instance_convert(forest.options[:output], instances)

  forest.model = Dict()
  forest.model[:output] = forest.options[:output]
  forest.model[:instance_conversion_required] = instance_conversion_required
  forest.model[:label_conversion_required] = label_conversion_required

  # Set training-dependent options
  impl_options = forest.options[:impl_options]
  if impl_options[:num_subfeatures] == nothing
    num_subfeatures = size(instances, 2)
  else
    num_subfeatures = impl_options[:num_subfeatures]
  end
  # Build model
  forest.model[:model] = DT.build_forest(
    labels, 
    instances,
    num_subfeatures, 
    impl_options[:num_trees],
    impl_options[:partial_sampling]
  )
end

function transform!(forest::RandomForest, instances::Matrix)
  if forest.model[:output] == :regression &&
    forest.model[:instance_conversion_required]

    instances = promote(instances)[1]
  end

  predictions = DT.apply_forest(forest.model[:model], instances)

  if forest.model[:output] == :class && 
    forest.model[:label_conversion_required]
    
    predictions = map(x -> float(x), predictions)
    predictions = [promote(predictions...)...]
  end

  return predictions
end

# Adaboosted C4.5 decision stumps.
type DecisionStumpAdaboost <: Learner
  model
  options
  
  function DecisionStumpAdaboost(options=Dict())
    default_options = {
      # Target output.
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Number of boosting iterations.
        :num_iterations => 7
      }
    }
    new(nothing, nested_dict_merge(default_options, options))
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
  adaboost.model = {
    :ensemble => ensemble,
    :coefficients => coefficients
  }
end

function transform!(adaboost::DecisionStumpAdaboost, instances::Matrix)
  return DT.apply_adaboost_stumps(
    adaboost.model[:ensemble], adaboost.model[:coefficients], instances
  )
end

end # module
