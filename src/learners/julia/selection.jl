# Selection methods for ensemble learning.
module SelectionMethods

importall Orchestra.AbstractLearner
import Orchestra.Util: kfold

import Orchestra.Learners.DecisionTreeWrapper: PrunedTree
import Orchestra.Learners.DecisionTreeWrapper: RandomForest
import Orchestra.Learners.LIBSVMWrapper: SVM

export BestLearnerSelection, 
       train!, 
       predict!

# Selects best learner out of set.
# 
# <pre>
# default_options = {
#     # Metric to train against
#     # (:accuracy).
#     :metric => :accuracy,
#     # Function to return partitions of instance indices.
#     :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
#     # Function that selects the best learner by index.
#     # Arg learner_partition_scores is a (learner, partition) score matrix.
#     :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],          
#     # Candidate learners.
#     :learners => [PrunedTree(), SVM(), RandomForest()]
# }
# </pre>
type BestLearnerSelection <: Learner
    model
    options
    
    function BestLearnerSelection(options=Dict())
        default_options = {
            # Metric to train against
            # (:accuracy).
            :metric => :accuracy,
            # Function to return partitions of instance indices.
            :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
            # Function that selects the best learner by index.
            # Arg learner_partition_scores is a (learner, partition) score matrix.
            :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],          
            # Candidate learners.
            :learners => [PrunedTree(), SVM(), RandomForest()]
        }
        new(nothing, merge(default_options, options)) 
    end
end

function train!(bls::BestLearnerSelection, instances::Matrix, labels::Vector)
    # Generate partitions
    partition_generator = bls.options[:partition_generator]
    partitions = partition_generator(instances, labels)
    
    # Train each learner on each partition and obtain validation metric
    learners = bls.options[:learners]
    num_partitions = size(partitions, 1)
    num_learners = size(learners, 1)
    num_instances = size(instances, 1)
    learner_partition_scores = Array(Any, num_learners, num_partitions)
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
        result = score(
            learner, validation_instances, validation_labels, predictions
        )
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
        :learners => learners,
        :learner_partition_scores => learner_partition_scores
    }
end

function predict!(bls::BestLearnerSelection, instances::Matrix)
    predict!(bls.model[:best_learner], instances)
end

end # module
