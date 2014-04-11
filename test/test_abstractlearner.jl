module TestAbstractLearner

using FactCheck
using Fixtures

using Orchestra.AbstractLearner

include(joinpath("learners", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

facts("Orchestra Learner functions", using_fixtures) do
  context("score calculates accuracy", using_fixtures) do
    learner = PerfectScoreLearner({:problem => nfcp})
    predictions = train_and_predict!(learner, nfcp)

    @fact score(
      learner, 
      nfcp.test_instances, nfcp.test_labels, predictions
    ) => 100.0
  end
  context("score throws exception on unknown metric", using_fixtures) do
    learner = PerfectScoreLearner({:metric => :fake, :problem => nfcp})
    predictions = train_and_predict!(learner, nfcp)

    @fact_throws score(
      learner, 
      nfcp.test_instances, nfcp.test_labels, predictions
    )
  end
end

end # module
