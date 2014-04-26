module TestOrchestraTransformers

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Transformers.OrchestraTransformers

facts("Orchestra transformers", using_fixtures) do
  context("OneHotEncoder transforms nominal features", using_fixtures) do
    instances = [
      2 "a" 1 "c";
      1 "b" 2 "d";
    ]
    labels = [
      "y";
      "z";
    ]
    expected_transformed = [
      2 1 0 1 1 0;
      1 0 1 2 0 1;
    ]
    encoder = OneHotEncoder()
    fit!(encoder, instances, labels)
    transformed = transform!(encoder, instances)

    @fact transformed => expected_transformed
  end

  context("OneHotEncoder transforms with options", using_fixtures) do
    nominal_columns = [2, 4]
    nominal_column_values_map = {
      2 => ["a", "b", "c", "d", "e"],
      4 => ["a", "b", "c", "d", "e"]
    }
    encoder = OneHotEncoder({
      :nominal_columns => nominal_columns,
      :nominal_column_values_map => nominal_column_values_map
    })
    fit!(encoder, fcp.train_instances, fcp.train_labels)
    transformed = transform!(encoder, fcp.test_instances)

    @fact size(transformed, 2) => 12
  end
end

end # module
