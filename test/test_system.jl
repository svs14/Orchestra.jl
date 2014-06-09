# System tests.
module TestSystem

using Orchestra.Types
using Orchestra.System
using Orchestra.Transformers
importall Orchestra.Util

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()
fcp = FeatureClassification()

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
concrete_transformer_types = setdiff(
  all_concrete_subtypes(Transformer),
  all_concrete_subtypes(TestLearner)
)

using FactCheck
using Fixtures

facts("Orchestra system", using_fixtures) do
  context("All transformers handle fixture data.", using_fixtures) do
    for concrete_transformer_type in concrete_transformer_types
      transformer = concrete_transformer_type()
      transformations = fit_and_transform!(transformer, nfcp)
      @fact typeof(transformations) <: Array => true
    end
  end

  context("All learners train and predict on fixture data.", using_fixtures) do
    for concrete_learner_type in concrete_learner_types
      learner = concrete_learner_type()
      predictions = fit_and_transform!(learner, nfcp)
      @fact infer_eltype(predictions) <: String => true
    end
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
      fit!(learner, train_instances, train_labels)
      predictions = transform!(learner, test_instances)
      @fact infer_eltype(predictions) <: String => true
    end
  end

  context("Ensemble with learners from different libraries work.", using_fixtures) do 
    learners = Learner[]
    push!(learners, RandomForest())
    push!(learners, StackEnsemble())
    if HAS_SKL
      push!(learners, SKLLearner())
    end
    if HAS_CRT
      push!(learners, CRTLearner())
    end
    ensemble = VoteEnsemble({:learners => learners})
    predictions = fit_and_transform!(ensemble, nfcp)

    @fact infer_eltype(predictions) <: String => true
  end

  context("Pipeline works with fixture data.", using_fixtures) do
    transformers = [
      OneHotEncoder(),
      Imputer(),
      StandardScaler(),
      BestLearner()
    ]
    pipeline = Pipeline({:transformers => transformers})
    predictions = fit_and_transform!(pipeline, fcp)

    @fact infer_eltype(predictions) <: String => true
  end
end

end # module
