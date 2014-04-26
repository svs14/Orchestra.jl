module TestOrchestraTransformers

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()
nfcp = NumericFeatureClassification()

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
    nominal_columns = [3, 5]
    nominal_column_values_map = {
      3 => ["a", "b", "c", "d", "e"],
      5 => ["a", "b", "c", "d", "e"]
    }
    encoder = OneHotEncoder({
      :nominal_columns => nominal_columns,
      :nominal_column_values_map => nominal_column_values_map
    })
    fit!(encoder, fcp.train_instances, fcp.train_labels)
    transformed = transform!(encoder, fcp.test_instances)

    @fact size(transformed, 2) => 13
  end

  context("Imputer replaces NA", using_fixtures) do
    instances = [
      1.0      1.0;
      nan(1.0) 1.0;
      0.0      0.0;
      nan(0.0) 0.0;
    ]
    labels = ["x";"x";"y";"y"]
    expected_transformed = [
      1.0 1.0;
      0.5 1.0;
      0.0 0.0;
      0.5 0.0;
    ]
    imputer = Imputer()
    fit!(imputer, instances, labels)
    transformed = transform!(imputer, instances)

    @fact transformed => expected_transformed
  end

  context("Pipeline chains transformers", using_fixtures) do
    pipe = Pipeline()
    fit!(pipe, fcp.train_instances, fcp.train_labels)
    transformed = transform!(pipe, fcp.test_instances)

    @fact size(transformed, 2) => 11
    @fact true => !any(map(x -> isnan(x), transformed))
  end

  context("Wrapper delegates to transformer", using_fixtures) do
    wrapper = Wrapper({
      :transformer => OneHotEncoder()
    })
    fit!(wrapper, fcp.train_instances, fcp.train_labels)
    transformed = transform!(wrapper, fcp.test_instances)

    @fact size(transformed, 2) => 11
  end

  context("Identity returns instances as is", using_fixtures) do
    id = Identity()
    fit!(id, fcp.train_instances, nfcp.train_labels)
    transformed = transform!(id, nfcp.test_instances)
    expected_transformed = nfcp.test_instances

    @fact transformed => expected_transformed
  end
end

end # module
