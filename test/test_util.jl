module TestUtil

using FactCheck
using Fixtures

using Orchestra.Util

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()

facts("Orchestra util functions", using_fixtures) do
  context("holdout returns proportional partitions", using_fixtures) do
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)

    @fact size(left, 1) => n - (n * right_prop)
    @fact size(right, 1) => n * right_prop
    @fact intersect(left, right) => isempty
    @fact size(union(left, right), 1) => n
  end
  context("kfold returns k partitions", using_fixtures) do
    num_instances = 10
    num_partitions = 3
    partitions = kfold(num_instances, num_partitions)

    @fact size(partitions, 1) => num_partitions
    # Check pairwise intersection of partitions
    @fact size([partitions...], 1) => size(unique([partitions...]), 1)
    @fact size(union(partitions...), 1) => num_instances
  end
  context("score calculates accuracy", using_fixtures) do
    learner = PerfectScoreLearner({:problem => nfcp})
    predictions = train_and_predict!(learner, nfcp)

    @fact score(
      :accuracy, nfcp.test_labels, predictions
    ) => 100.0
  end
  context("score throws exception on unknown metric", using_fixtures) do
    learner = PerfectScoreLearner({:problem => nfcp})
    predictions = train_and_predict!(learner, nfcp)

    @fact_throws score(
      :fake, nfcp.test_labels, predictions
    )
  end
end

end # module
