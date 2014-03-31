# Decision trees as found in DecisionTree Julia package.
module DecisionTreeWrapper

importall Orchestra.AbstractLearner
import DecisionTree
dt = DecisionTree

export PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       train!, 
       predict!

# Pruned ID3 decision tree.
# 
# <pre>
# default_options = {
#     # Metric to train against
#     # (:accuracy).
#     :metric => :accuracy,
#     # Options specific to this implementation.
#     :impl_options => {
#         # Merge leaves having >= purity_threshold combined purity.
#         :purity_threshold => 1.0
#     },
# }
# </pre>
type PrunedTree <: Learner
    model
    options
    
    function PrunedTree(options=Dict())
        default_options = {
            # Metric to train against
            # (:accuracy).
            :metric => :accuracy,
            # Options specific to this implementation.
            :impl_options => {
                # Merge leaves having >= purity_threshold combined purity.
                :purity_threshold => 1.0
            },
        }
        new(nothing, merge(default_options, options))
    end
end

function train!(tree::PrunedTree, instances::Matrix, labels::Vector)
    impl_options = tree.options[:impl_options]
    tree.model = dt.build_tree(labels, instances)
    tree.model = dt.prune_tree(tree.model,impl_options[:purity_threshold])
end
function predict!(tree::PrunedTree, instances::Matrix)
    return dt.apply_tree(tree.model, instances)
end

# Random forest (C4.5).
#
# <pre>
# default_options = {
#     # Metric to train against
#     # (:accuracy).
#     :metric => :accuracy,
#     # Options specific to this implementation.
#     :impl_options => {
#         # Number of features to train on with trees.
#         :num_subfeatures => nothing,
#         # Number of trees in forest.
#         :num_trees => 10,
#         # Proportion of trainingset to be used for trees.
#         :partial_sampling => 0.7
#     },
# }
# </pre>
type RandomForest <: Learner
    model
    options
    
    function RandomForest(options=Dict())
        default_options = {
            # Metric to train against
            # (:accuracy).
            :metric => :accuracy,
            # Options specific to this implementation.
            :impl_options => {
                # Number of features to train on with trees.
                :num_subfeatures => nothing,
                # Number of trees in forest.
                :num_trees => 10,
                # Proportion of trainingset to be used for trees.
                :partial_sampling => 0.7
            },
        }
        new(nothing, merge(default_options, options))
    end
end

function train!(forest::RandomForest, instances::Matrix, labels::Vector)
    # Set training-dependent options
    impl_options = forest.options[:impl_options]
    if impl_options[:num_subfeatures] == nothing
        num_subfeatures = size(instances, 2)
    else
        num_subfeatures = impl_options[:num_subfeatures]
    end
    # Build model
    forest.model = dt.build_forest(
        labels, 
        instances,
        num_subfeatures, 
        impl_options[:num_trees],
        impl_options[:partial_sampling]
    )
end

function predict!(forest::RandomForest, instances::Matrix)
    return dt.apply_forest(forest.model, instances)
end

# Adaboosted C4.5 decision stumps.
# 
# <pre>
# default_options = {
#     # Metric to train against
#     # (:accuracy).
#     :metric => :accuracy,
#     # Options specific to this implementation.
#     :impl_options => {
#         # Number of boosting iterations.
#         :num_iterations => 7
#     },
# }
# </pre>
type DecisionStumpAdaboost <: Learner
    model
    coefficients
    options
    
    function DecisionStumpAdaboost(options=Dict())
        default_options = {
            # Metric to train against
            # (:accuracy).
            :metric => :accuracy,
            # Options specific to this implementation.
            :impl_options => {
                # Number of boosting iterations.
                :num_iterations => 7
            },
        }
        new(nothing, nothing, merge(default_options, options))
    end
end

function train!(adaboost::DecisionStumpAdaboost, 
    instances::Matrix, labels::Vector)

    adaboost.model, adaboost.coefficients = dt.build_adaboost_stumps(
        labels, instances, adaboost.options[:impl_options][:num_iterations]
    )
end

function predict!(adaboost::DecisionStumpAdaboost, instances::Matrix)
    return dt.apply_adaboost_stumps(
        adaboost.model, adaboost.coefficients, instances
    )
end

end #module
