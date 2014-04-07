module TestSelectionMethods

include("../fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.SelectionMethods

facts("Selection learners", using_fixtures) do
  context("BestLearnerSelection picks the best learner", using_fixtures) do
    always_a_options = { :label => "a" }
    always_b_options = { :label => "b" }
    learner = BestLearnerSelection({
      :learners => [
        AlwaysSameLabelLearner(always_a_options),
        PerfectScoreLearner(),
        AlwaysSameLabelLearner(always_b_options)
      ]
    })
    train!(learner, train_instances, train_labels)

    @fact learner.model[:best_learner_index] => 2
  end
end

end # module
