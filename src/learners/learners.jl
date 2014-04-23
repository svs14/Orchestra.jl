# Learner definitions and implementations.
module Learners

export Learner,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       VoteEnsemble, 
       StackEnsemble,
       BestLearnerSelection,
       SKLWrapper,
       CRTWrapper,
       train!,
       predict!,
       score

# Obtain system details
import Orchestra.System: HAS_SKL, HAS_CRT

# Include abstract learner as convenience
importall Orchestra.AbstractLearner

# Include atomic Julia learners
include(joinpath("julia", "decisiontree.jl"))
importall .DecisionTreeWrapper

# Include atomic Python learners
if HAS_SKL
  include(joinpath("python", "scikit_learn.jl"))
  importall .ScikitLearnWrapper
end

# Include atomic R learners
if HAS_CRT
  include(joinpath("r", "caret.jl"))
  importall .CaretWrapper
end

# Include aggregate learners last, dependent on atomic learners
include(joinpath("orchestra", "ensemble.jl"))
importall .EnsembleMethods

end # module
