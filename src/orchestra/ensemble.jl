# Ensemble learning methods.
module EnsembleMethods

importall Orchestra.AbstractLearner
import Orchestra.Util: holdout, kfold, score
import Stats
import MLBase

import Orchestra.Learners.DecisionTreeWrapper: train!, predict!
import Orchestra.Learners.DecisionTreeWrapper: PrunedTree
import Orchestra.Learners.DecisionTreeWrapper: RandomForest
import Orchestra.Learners.DecisionTreeWrapper: DecisionStumpAdaboost

export VoteEnsemble, 
       StackEnsemble,
       BestLearnerEnsemble, 
       train!, 
       predict!

# Set of machine learners that majority vote to decide prediction.
#
# <pre>
# default_options = {
#   # Output to train against
#   # (:class).
#   :output => :class,
#   # Learners in voting committee.
#   :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
# }
# </pre>
type VoteEnsemble <: Learner
  model
  options
  
  function VoteEnsemble(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Learners in voting committee.
      :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
    }
    new(nothing, merge(default_options, options))
  end
end

function train!(ve::VoteEnsemble, instances::Matrix, labels::Vector)
  # Train all learners
  learners = ve.options[:learners]
  for learner in learners
    train!(learner, instances, labels)
  end
  ve.model = { :learners => learners }
end

function predict!(ve::VoteEnsemble, instances::Matrix)
  # Make learners vote
  learners = ve.options[:learners]
  predictions = map(learner -> predict!(learner, instances), learners)
  # Return majority vote prediction
  return Stats.mode(predictions)
end

# Ensemble where a 'stack' learner learns on a set of learners' predictions.
#
# <pre>
# default_options = {    
#   # Output to train against
#   # (:class).
#   :output => :class,
#   # Set of learners that produce feature space for stacker.
#   :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
#   # Machine learner that trains on set of learners' outputs.
#   :stacker => RandomForest(),
#   # Proportion of training set left to train stacker itself.
#   :stacker_training_proportion => 0.3,
#   # Provide original features on top of learner outputs to stacker.
#   :keep_original_features => false
# }
# </pre>
type StackEnsemble <: Learner
  model
  options
  
  function StackEnsemble(options=Dict())
    default_options = {    
      # Output to train against
      # (:class).
      :output => :class,
      # Set of learners that produce feature space for stacker.
      :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
      # Machine learner that trains on set of learners' outputs.
      :stacker => RandomForest(),
      # Proportion of training set left to train stacker itself.
      :stacker_training_proportion => 0.3,
      # Provide original features on top of learner outputs to stacker.
      :keep_original_features => false
    }
    new(nothing, merge(default_options, options)) 
  end
end

function train!(se::StackEnsemble, instances::Matrix, labels::Vector)
  learners = se.options[:learners]
  num_learners = size(learners, 1)
  num_instances = size(instances, 1)
  num_labels = size(labels, 1)
  
  # Perform holdout to obtain indices for 
  # partitioning learner and stacker training sets
  shuffled_indices = randperm(num_instances)
  stack_proportion = se.options[:stacker_training_proportion]
  (learner_indices, stack_indices) = holdout(num_instances, stack_proportion)
  
  # Partition training set for learners and stacker
  learner_instances = instances[learner_indices, :]
  stack_instances = instances[stack_indices, :]
  learner_labels = labels[learner_indices]
  stack_labels = labels[stack_indices]
  
  # Train all learners
  for learner in learners
    train!(learner, learner_instances, learner_labels)
  end
  
  # Train stacker on learners' outputs
  label_map = MLBase.labelmap(labels)
  stacker = se.options[:stacker]
  keep_original_features = se.options[:keep_original_features]
  stacker_instances = build_stacker_instances(
    learners, stack_instances, label_map, keep_original_features
  )
  train!(stacker, stacker_instances, stack_labels)
  
  # Build model
  se.model = {
    :learners => learners, 
    :stacker => stacker, 
    :label_map => label_map, 
    :keep_original_features => keep_original_features
  }
end

function predict!(se::StackEnsemble, instances::Matrix)
  # Build stacker instances
  learners = se.model[:learners]
  stacker = se.model[:stacker]
  label_map = se.model[:label_map]
  keep_original_features = se.model[:keep_original_features]
  stacker_instances = build_stacker_instances(
    learners, instances, label_map, keep_original_features
  )

  # Predict
  return predict!(stacker, stacker_instances)
end

# Build stacker instances.
function build_stacker_instances{T<:Learner}(
  learners::Vector{T}, instances::Matrix, 
  label_map, keep_original_features=false)

  # Build empty stacker instance space
  num_labels = size(label_map.vs, 1)
  num_instances = size(instances, 1)
  num_learners = size(learners, 1)
  stacker_instances = zeros(num_instances, num_learners * num_labels)

  # Fill stack instances with predictions from learners
  for l_index = 1:num_learners
    predictions = predict!(learners[l_index], instances)
    for p_index in 1:size(predictions, 1)
      pred_encoding = MLBase.labelencode(label_map, predictions[p_index])
      pred_column = (l_index-1) * num_labels + pred_encoding
      stacker_instances[p_index, pred_column] = 
        one(eltype(stacker_instances))
    end
  end

  # Add original features to stacker instance space if enabled
  if keep_original_features
    stacker_instances = [instances stacker_instances]
  end
  
  # Return stacker instances
  return stacker_instances
end

# Selects best learner out of set.
# 
# <pre>
# default_options = {
#   # Output to train against
#   # (:class).
#   :output => :class,
#   # Function to return partitions of instance indices.
#   :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
#   # Function that selects the best learner by index.
#   # Arg learner_partition_scores is a (learner, partition) score matrix.
#   :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],      
#   # Score type returned by score() using respective output.
#   :score_type => Real
#   # Candidate learners.
#   :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
# }
# </pre>
type BestLearnerEnsemble <: Learner
  model
  options
  
  function BestLearnerEnsemble(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Function to return partitions of instance indices.
      :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
      # Function that selects the best learner by index.
      # Arg learner_partition_scores is a (learner, partition) score matrix.
      :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],      
      # Score type returned by score() using respective output.
      :score_type => Real,
      # Candidate learners.
      :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
    }
    new(nothing, merge(default_options, options)) 
  end
end

function train!(bls::BestLearnerEnsemble, instances::Matrix, labels::Vector)
  # Generate partitions
  partition_generator = bls.options[:partition_generator]
  partitions = partition_generator(instances, labels)
  
  # Train each learner on each partition and obtain validation output
  learners = bls.options[:learners]
  num_partitions = size(partitions, 1)
  num_learners = size(learners, 1)
  num_instances = size(instances, 1)
  score_type = bls.options[:score_type]
  learner_partition_scores = Array(score_type, num_learners, num_partitions)
  for l_index = 1:num_learners, p_index = 1:num_partitions
    partition = partitions[p_index]
    rest = setdiff(1:num_instances, partition)
    learner = learners[l_index]

    training_instances = instances[partition,:]
    training_labels = labels[partition]
    validation_instances = instances[rest, :]
    validation_labels = labels[rest]

    train!(learner, training_instances, training_labels)
    predictions = predict!(learner, validation_instances)
    result = score(:accuracy, validation_labels, predictions)
    learner_partition_scores[l_index, p_index] = result
  end
  
  # Find best learner based on selection function
  best_learner_index = 
    bls.options[:selection_function](learner_partition_scores)
  best_learner = learners[best_learner_index]
  
  # Retrain best learner on all training instances
  train!(best_learner, instances, labels)
  
  # Create model
  bls.model = {
    :best_learner => best_learner,
    :best_learner_index => best_learner_index,
    :learners => learners,
    :learner_partition_scores => learner_partition_scores
  }
end

function predict!(bls::BestLearnerEnsemble, instances::Matrix)
  predict!(bls.model[:best_learner], instances)
end

end # module
