module TestEnsembleMethods

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Learners.EnsembleMethods

facts("Ensemble learners", using_fixtures) do
  context("VoteEnsemble predicts according to majority", using_fixtures) do
    always_a_options = { :label => "a" }
    always_b_options = { :label => "b" }
    learner = VoteEnsemble({
      :learners => [
        AlwaysSameLabelLearner(always_a_options),
        AlwaysSameLabelLearner(always_a_options),
        AlwaysSameLabelLearner(always_b_options)
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)
    predictions = transform!(learner, nfcp.test_instances)
    expected_predictions = fill("a", size(nfcp.test_instances, 1))

    @fact predictions => expected_predictions
  end

  context("StackEnsemble predicts with combined learners", using_fixtures) do
    # Fix random seed, due to stochasticity in stacker.
    srand(2)

    always_a_options = { :label => "a" }
    learner = StackEnsemble({
      :learners => [
        AlwaysSameLabelLearner(always_a_options),
        AlwaysSameLabelLearner(always_a_options),
        PerfectScoreLearner()
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)
    predictions = transform!(learner, nfcp.test_instances)
    unexpected_predictions = fill("a", size(nfcp.test_instances, 1))

    @fact predictions => not(unexpected_predictions)
  end

  context("BestLearnerEnsemble picks the best learner", using_fixtures) do
    always_a_options = { :label => "a" }
    always_b_options = { :label => "b" }
    learner = BestLearnerEnsemble({
      :learners => [
        AlwaysSameLabelLearner(always_a_options),
        PerfectScoreLearner(),
        AlwaysSameLabelLearner(always_b_options)
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)

    @fact learner.model[:best_learner_index] => 2
  end
end

end # module
