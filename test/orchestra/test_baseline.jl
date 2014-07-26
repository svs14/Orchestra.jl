module TestBaselineMethods

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = MLProblem(;
  output = :class,
  feature_type = Float64,
  label_type = Float64,
  handle_na = false,
  dataset_type = Matrix{Float64}
)

using FactCheck


importall Orchestra.Transformers.BaselineMethods

facts("Baseline transformers") do
  context("Baseline learner does simple transforms") do
    bl = Baseline()
    instances = Float64[
      1 1;
      1 2;
      2 2;
    ]
    labels = Float64[1.0; 1.0; 2.0]
    fit!(bl, instances, labels)
    transformed = transform!(bl, instances)
    expected_transformed = Float64[1.0; 1.0; 1.0]

    @fact transformed => expected_transformed
  end
  context("Identity returns instances as is") do
    id = Identity()
    fit!(id, nfcp.train_instances, nfcp.train_labels)
    transformed = transform!(id, nfcp.test_instances)
    expected_transformed = nfcp.test_instances

    @fact transformed => expected_transformed
  end
end

end # module
