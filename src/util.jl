# Various functions that work with learners.
module Util

import MLBase: Kfold

export holdout,
       kfold

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

end # module
