# Abstract learner types and methods.
module AbstractLearner

export Learner,
       TestLearner,
       train!,
       predict!

# Learner abstract type which all machine learners implement.
# All learner types must have implementations 
# of function `train!` and `predict!`.
abstract Learner

# Test learner. 
# Used to separate production learners from test.
abstract TestLearner <: Learner

# Trains learner on provided instances and labels.
#
# @param learner Target machine learner.
# @param instances Training instances.
# @param labels Training labels.
function train!(learner::Learner, instances::Matrix, labels::Vector)
  error(typeof(learner), " does not implement train!")
end

# Trains learner on provided instances and labels.
#
# @param learner Target machine learner.
# @param instances Candidate instances.
# @return Predictions.
function predict!(learner::Learner, instances::Matrix)
  error(typeof(learner), " does not implement predict!")
end

end # module
