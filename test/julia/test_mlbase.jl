module TestMLBaseWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Transformers.MLBaseWrapper

facts("MLBase transformers", using_fixtures) do
  context("Standardizer transforms features", using_fixtures) do
    instances = [
      5 10;
      -5 0;
      0 5;
    ]
    labels = [
      "x";
      "y";
      "z";
    ]
    expected_transformed = [
      1.0 1.0;
      -1.0 -1.0;
      0.0 0.0;
    ]
    standardizer = Standardizer()
    fit!(standardizer, instances, labels)
    transformed = transform!(standardizer, instances)

    @fact transformed => expected_transformed
  end
end

end # module
