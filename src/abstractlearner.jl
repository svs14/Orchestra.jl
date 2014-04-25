# Abstract learner types and methods.
module AbstractLearner

export Learner,
       TestLearner,
       fit!,
       transform!

# Learner abstract type which all machine learners implement.
# All learner types must have implementations 
# of function `fit!` and `transform!`.
abstract Learner

# Test learner. 
# Used to separate production learners from test.
abstract TestLearner <: Learner

# Trains learner on provided instances and labels.
#
# @param learner Target machine learner.
# @param instances Training instances.
# @param labels Training labels.
function fit!(learner::Learner, instances::Matrix, labels::Vector)
  error(typeof(learner), " does not implement fit!")
end

# Trains learner on provided instances and labels.
#
# @param learner Target machine learner.
# @param instances Candidate instances.
# @return Predictions.
function transform!(learner::Learner, instances::Matrix)
  error(typeof(learner), " does not implement transform!")
end

end # module
