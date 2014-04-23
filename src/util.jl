# Various functions that work with learners.
module Util

import MLBase: Kfold
using Match

export holdout,
       kfold,
       score

# Holdout method that partitions a collection
# into two partitions.
#
# @param n Size of collection to partition.
# @param right_prop Percentage of collection placed in right partition.
# @return Two partitions of indices, left and right.
function holdout(n, right_prop)
  shuffled_indices = randperm(n)
  partition_pivot = int(right_prop * n)
  right = shuffled_indices[1:partition_pivot]
  left = shuffled_indices[partition_pivot+1:end]
  return (left, right)
end

# Returns k-fold partitions.
#
# @param num_instances Total number of instances.
# @param num_partitions Number of partitions required.
# @return Returns num_partitions-element Array{Any, 1} with
#     partition of indices as elements.
function kfold(num_instances, num_partitions)
  return collect(Kfold(num_instances, num_partitions))
end

# Score learner predictions against ground truth values.
#
# Available metrics:
# - :accuracy
#
# @param metric Metric to assess with.
# @param actual Ground truth values.
# @param predicted Predicted values.
# @return Score of learner.
function score(metric::Symbol, actual, predicted)
  return @match metric begin
    :accuracy => mean(actual .== predicted) * 100.0
    _ => error("Metric $metric not implemented for score.")
  end
end

end # module
