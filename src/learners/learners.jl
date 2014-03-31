# Learner definitions and implementations.
module Learners

export Learner,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       VoteEnsemble, 
       StackEnsemble,
       SVM,
       BestLearnerSelection,
       train!,
       predict!,
       score

# Include abstract learner as convenience
using Orchestra.AbstractLearner

# Include atomic Julia learners
include(joinpath("julia", "decisiontree.jl"))
using .DecisionTreeWrapper
include(joinpath("julia", "libsvm.jl"))
using .LIBSVMWrapper

# Include aggregate learners last, dependent on atomic learners
include(joinpath("julia", "ensemble.jl"))
using .EnsembleMethods
include(joinpath("julia", "selection.jl"))
using .SelectionMethods

end # module
