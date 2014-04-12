# System tests.
module TestSystem

using Orchestra.AbstractLearner
using Orchestra.System
using Orchestra.Learners
using Orchestra.Util

include(joinpath("learners", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

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

concrete_learner_types = setdiff(
  all_concrete_subtypes(Learner),
  all_concrete_subtypes(TestLearner)
)

using FactCheck
using Fixtures

facts("Orchestra system", using_fixtures) do
  context("All learners train and predict on fixture data.", using_fixtures) do

    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      train_and_predict!(learner, nfcp)
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
    learners = Learner[]
    push!(learners, RandomForest())
    push!(learners, StackEnsemble())
    if HAS_SKL
      push!(learners, SKLWrapper())
    end
    if HAS_CRT
      push!(learners, CRTWrapper())
    end
    ensemble = VoteEnsemble({:learners => learners})
    predictions = train_and_predict!(ensemble, nfcp)

    @fact 1 => 1
  end
end

end # module
