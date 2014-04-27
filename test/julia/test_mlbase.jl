module TestMLBaseWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Transformers.MLBaseWrapper

facts("MLBase transformers", using_fixtures) do
  context("StandardScaler transforms features", using_fixtures) do
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
    standard_scaler = StandardScaler()
    fit!(standard_scaler, instances, labels)
    transformed = transform!(standard_scaler, instances)

    @fact transformed => expected_transformed
  end
end

end # module
