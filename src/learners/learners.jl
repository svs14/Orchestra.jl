# Learner definitions and implementations.
module Learners

import PyCall: pyimport, pycall

export Learner,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       VoteEnsemble, 
       StackEnsemble,
       BestLearnerSelection,
       SKLLearner,
       SKLRandomForest,
       SKLExtraTrees,
       SKLGradientBoosting,
       SKLLogisticRegression,
       SKLPassiveAggressive,
       SKLRidge,
       SKLRidgeCV,
       SKLSGD,
       SKLKNeighbors,
       SKLRadiusNeighbors,
       SKLNearestCentroid,
       SKLSVC,
       SKLLinearSVC,
       SKLNuSVC,
       SKLDecisionTree,
       CRTWrapper,
       HAS_SKL,
       HAS_CRL,
       train!,
       predict!,
       score

function check_py_dep(package::String)
  is_available = true
  try
    pyimport(package)
  catch
    is_available = false
  end
  return is_available
end

function check_r_dep(package::String)
  is_available = true
  try
    rpy2_packages = pyimport("rpy2.robjects.packages")
    pycall(rpy2_packages["importr"], Any, package)
  catch
    is_available = false
  end
  return is_available
end

# Check system for python dependencies.
HAS_SKL = check_py_dep("sklearn")
HAS_CRT = check_r_dep("caret")

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
include(joinpath("orchestra", "selection.jl"))
importall .SelectionMethods

end # module
