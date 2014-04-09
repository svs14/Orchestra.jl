# System tests.
# 
# NOTE(svs14): This must be run outside of runner.jl
#              to avoid loading test learners.
module TestSystem

using Orchestra.Learners
using Orchestra.Util

function all_concrete_subtypes(a_type::Type)
  a_subtypes = Type[]
  for a_subtype in subtypes(a_type)
    if isleaftype(a_subtype)
      push!(a_subtypes, a_subtype)
    else
      append!(a_subtypes, all_concrete_subtypes(a_subtype))
    end
  end
  return a_subtypes
end
# NOTE(svs14): This must be called before FixtureLearners is loaded.
#              Otherwise test learners will be included.
concrete_learner_types = all_concrete_subtypes(Learner)

include(joinpath("learners", "fixture_learners.jl"))
import .FixtureLearners
FL = FixtureLearners

using FactCheck
using Fixtures

facts("Orchestra system", using_fixtures) do
  context("All learners train and predict on fixture data.", using_fixtures) do

    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      FL.train_and_predict!(learner)
    end

    @fact 1 => 1
  end

  context("All learners train and predict on iris dataset.", using_fixtures) do
    # Get data
    dataset = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
    instances = dataset[:,1:(end-1)]
    labels = dataset[:, end]
    (train_ind, test_ind) = holdout(size(instances, 1), 0.3)
    train_instances = instances[train_ind, :]
    test_instances = instances[test_ind, :]
    train_labels = labels[train_ind]
    test_labels = labels[test_ind]

    # Test all learners
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      train!(learner, train_instances, train_labels)
      predict!(learner, test_instances)
    end

    @fact 1 => 1
  end

  context("Ensemble with learners from different libraries work.", using_fixtures) do 
    learners = [
      RandomForest(), 
      StackEnsemble(),
      SKLSVC(),
      CRTWrapper()
    ]
    ensemble = VoteEnsemble({:learners => learners})
    predictions = FL.train_and_predict!(ensemble)

    @fact 1 => 1
  end
end

end # module
