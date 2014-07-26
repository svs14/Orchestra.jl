module TestMLBaseWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = MLProblem(;
  output = :class,
  feature_type = Any,
  label_type = Any,
  handle_na = true,
  dataset_type = Matrix
)

using FactCheck


importall Orchestra.Transformers.MLBaseWrapper

facts("MLBase transformers") do
  context("StandardScaler transforms features") do
    instances = Float64[
      5 10;
      -5 0;
      0 5;
    ]
    labels = Float64[
      1.0;
      2.0;
      3.0;
    ]
    expected_transformed = Float64[
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
