# Abstract learner types and methods.
module AbstractLearner

using Match

export Learner,
       train!,
       predict!,
       score

# Learner abstract type which all machine learners implement.
# All learner types must have implementations 
# of function `train!` and `predict!`.
abstract Learner

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

# Score learner predictions against ground truth labels.
# Learner must have learner.options[:metric] assigned to a metric.
#
# Current metrics are:
#   :accuracy
#
# @param learner Target machine learner.
# @param instances Instances to be scored on.
# @param instances Ground truth labels.
# @param instances Predicted labels.
# @return Score of learner.
function score(learner::Learner, instances::Matrix, labels::Vector,
  predictions::Vector)

  metric = learner.options[:metric]
  return @match metric begin
    :accuracy => mean(labels .== predictions) * 100.0
    _ => error("Metric $metric not implemented for score.")
  end
end

end # module
