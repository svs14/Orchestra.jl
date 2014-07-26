module TestOrchestraTransformers

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners

using FactCheck


importall Orchestra.Transformers.OrchestraTransformers

facts("Orchestra transformers") do
  context("OneHotEncoder transforms with nominal columns option") do
    instances = Float64[
      2 1 1 3;
      1 2 2 4;
    ]
    labels = Float64[
      1;
      2;
    ]
    expected_transformed = Float64[
      2 1 0 1 1 0;
      1 0 1 2 0 1;
    ]
    encoder = OneHotEncoder({:nominal_columns => [2, 4]})
    fit!(encoder, instances, labels)
    transformed = transform!(encoder, instances)

    @fact transformed => expected_transformed
  end
  context("OneHotEncoder transforms with all options") do
    instances = Float64[
      2 1 1 3;
      1 2 2 4;
    ]
    labels = Float64[
      1;
      2;
    ]
    expected_transformed = Float64[
      2 1 0 0 1 1 0;
      1 0 1 0 2 0 1;
    ]

    nominal_columns = [2, 4]
    nominal_column_values_map = {
      2 => [1, 2, 3],
      4 => [3, 4]
    }
    encoder = OneHotEncoder({
      :nominal_columns => nominal_columns,
      :nominal_column_values_map => nominal_column_values_map
    })
    fit!(encoder, instances, labels)
    transformed = transform!(encoder, instances)

    @fact transformed => expected_transformed
  end
  context("OneHotEncoder handles without options") do
    instances = Float64[
      2 1 1 3;
      1 2 2 4;
    ]
    labels = Float64[
      1;
      2;
    ]
    expected_transformed = Float64[
      2 1 1 3;
      1 2 2 4;
    ]

    encoder = OneHotEncoder()
    fit!(encoder, instances, labels)
    transformed = transform!(encoder, instances)

    @fact transformed => expected_transformed
  end

  context("Imputer replaces NA") do
    instances = Float64[
      1.0      1.0;
      nan(1.0) 1.0;
      0.0      0.0;
      nan(0.0) 0.0;
    ]
    labels = Float64[1; 1; 2; 2]
    expected_transformed = Float64[
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

  context("Pipeline chains transformers") do
    instances = Float64[
      1.0      1.0;
      nan(1.0) 1.0;
      0.0      0.0;
      nan(0.0) 0.0;
    ]
    labels = Float64[1; 1; 2; 2]
    expected_transformed = Float64[
      1.0 1.0;
      0.5 1.0;
      0.0 0.0;
      0.5 0.0;
    ]

    pipe = Pipeline()
    fit!(pipe, instances, labels)
    transformed = transform!(pipe, instances)

    @fact transformed => expected_transformed
    @fact !any([isnan(x) for x in transformed]) => true
  end

  context("Wrapper delegates to transformer") do
    instances = Float64[
      2 1 1 3;
      1 2 2 4;
    ]
    labels = Float64[
      1;
      2;
    ]
    expected_transformed = Float64[
      2 1 1 3;
      1 2 2 4;
    ]

    wrapper = Wrapper({
      :transformer => OneHotEncoder(),
      :transformer_options => Dict()
    })
    fit!(wrapper, instances, labels)
    transformed = transform!(wrapper, instances)

    @fact transformed => expected_transformed
  end
end

end # module
