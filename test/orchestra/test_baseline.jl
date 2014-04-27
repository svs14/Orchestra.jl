module TestBaselineMethods

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()
nfcp = NumericFeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Transformers.BaselineMethods

facts("Baseline transformers", using_fixtures) do
  context("Baseline learner does simple transforms", using_fixtures) do
    bl = Baseline()
    instances = [
      1 1;
      1 2;
      2 2;
    ]
    labels = ["a"; "a"; "b"]
    fit!(bl, instances, labels)
    transformed = transform!(bl, instances)
    expected_transformed = ["a"; "a"; "a"]

    @fact transformed => expected_transformed
  end
  context("Identity returns instances as is", using_fixtures) do
    id = Identity()
    fit!(id, nfcp.train_instances, nfcp.train_labels)
    transformed = transform!(id, nfcp.test_instances)
    expected_transformed = nfcp.test_instances

    @fact transformed => expected_transformed
  end
end

end # module
