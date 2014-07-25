module FixtureLearners

import DataFrames: DataArray, DataFrame, complete_cases
import MLBase: labelmap, labelencode

importall Orchestra.Types
importall Orchestra.Util
importall Orchestra.Conversion

export MLProblem,
       PerfectScoreLearner,
       AlwaysSameLabelLearner,
       fit_and_transform!,
       fit!,
       transform!

# NOTE(svs14): Currently hardcoded example. 
#              Consider turning into rule-based generator.
train_dataset = [
  1.0        1.0 "b"  2.0 "c" "a" -13.0;
  2.0        2.0 "b"  3.0 "c" "a" -9.0;
  nan(3.0)   3.0 "b"  4.0 "c" "a" -8.0;
  -1.0      -1.0 "d" -2.0 "c" "b" -3.0;
  -2.0      -2.0 "d" -3.0 "c" "b" 1.0;
  nan(-3.0) -3.0 "d" -4.0 "c" "b" 2.0;
  1.0        1.0 "a"  1.0 "a" "c" 17.0;
  2.0        2.0 "b"  2.0 "b" "c" 21.0;
  nan(3.0)   3.0 "c"  3.0 "c" "c" 22.0;
  0.0        0.0 "e"  1.0 "a" "d" 57.0;
  0.0        0.0 "e"  2.0 "b" "d" 61.0;
  nan(0.0)   0.0 "e"  3.0 "c" "d" 62.0;
]
test_dataset = [
  4.0        4.0 "b"  5.0 "c" "a" -11.0;
  nan(4.0)   5.0 "b"  6.0 "c" "a" -9.0;
  -4.0      -4.0 "d" -5.0 "c" "b" -1.0;
  nan(-4.0) -5.0 "d" -6.0 "c" "b" 1.0;
  4.0        4.0 "d"  4.0 "d" "c" 19.0;
  nan(4.0)   5.0 "e"  5.0 "e" "c" 21.0;
  0.0        0.0 "e"  4.0 "d" "d" 59.0;
  nan(0.0)   0.0 "e"  5.0 "e" "d" 61.0;
]

# Defines a given ML problem with associated fixture.
type MLProblem
  train_instances
  test_instances
  train_labels
  test_labels

  function MLProblem(;output=:class,
    feature_type=Any, label_type=Any,
    handle_na=false, dataset_type=Matrix{Float64})

    # Build instances
    train_instances = orchestra_convert(DataFrame, train_dataset[:, 1:end-2])
    test_instances = orchestra_convert(DataFrame, test_dataset[:, 1:end-2])
    if feature_type == Any
      nothing
    elseif feature_type == Float64
      train_instances = train_instances[:, [2, 4]]
      test_instances = test_instances[:, [2, 4]]
    end
    if !handle_na
      train_instances = train_instances[complete_cases(train_instances), :]
      test_instances = test_instances[complete_cases(test_instances), :]
    end

    # Build labels
    if output == :class
      if label_type == Any
        train_labels = orchestra_convert(DataArray, train_dataset[:, end-1])
        test_labels = orchestra_convert(DataArray, test_dataset[:, end-1])
      elseif label_type == Float64
        lm = labelmap(vcat(train_dataset[:, end-1], test_dataset[:, end-1]))
        train_labels = orchestra_convert(
          DataArray,
          labelencode(lm, train_dataset[:, end-1])
        )
        test_labels = orchestra_convert(
          DataArray,
          labelencode(lm, test_dataset[:, end-1])
        )
      end
    elseif output == :regression
      if label_type == Any
        train_labels = orchestra_convert(DataArray, train_dataset[:, end])
        test_labels = orchestra_convert(DataArray, test_dataset[:, end])
      elseif label_type == Float64
        train_labels = orchestra_convert(
          DataArray,
          convert(Vector{Float64}, train_dataset[:, end])
        )
        test_labels = orchestra_convert(
          DataArray,
          convert(Vector{Float64}, test_dataset[:, end])
        )
      end
    end

    # Convert dataset into required type
    if dataset_type == DataFrame
      nothing
    elseif dataset_type == Matrix{Float64}
      train_instances = orchestra_convert(Matrix{Float64}, train_instances)
      test_instances = orchestra_convert(Matrix{Float64}, test_instances)
      train_labels = orchestra_convert(Vector{Float64}, train_labels)
      test_labels = orchestra_convert(Vector{Float64}, test_labels)
    elseif dataset_type == Matrix
      train_instances = orchestra_convert(Matrix, train_instances)
      test_instances = orchestra_convert(Matrix, test_instances)
      train_labels = orchestra_convert(Vector, train_labels)
      test_labels = orchestra_convert(Vector, test_labels)
    end

    new(
      train_instances,
      test_instances,
      train_labels,
      test_labels
    ) 
  end
end

# Methods for transformations

function fit_and_transform!(transformer::Transformer, problem::MLProblem, seed=1)
  srand(seed)
  fit!(transformer, problem.train_instances, problem.train_labels)
  return transform!(transformer, problem.test_instances)
end

# Test Learners

type PerfectScoreLearner <: TestLearner
  model
  options

  function PerfectScoreLearner(options=Dict())
    nfcp = MLProblem(;
      output = :class,
      feature_type = Float64,
      label_type = Any,
      handle_na = false,
      dataset_type = Matrix
    )
    default_options = {
      :output => :class,
      :problem => nfcp
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(
  psl::PerfectScoreLearner, instances::Matrix, labels::Vector)

  problem = psl.options[:problem]

  dataset = [
    problem.train_instances problem.train_labels; 
    problem.test_instances problem.test_labels
  ]
  instance_label_map = [
    dataset[i,1:2] => dataset[i,3] for i=1:size(dataset, 1)
  ]

  psl.model = {
    :map => instance_label_map
  }
end

function transform!(
  psl::PerfectScoreLearner, instances::Matrix)

  num_instances = size(instances, 1)
  predictions = Array(String, num_instances)
  for i in 1:num_instances
    predictions[i] = psl.model[:map][instances[i,:]]
  end
  return predictions
end

type AlwaysSameLabelLearner <: TestLearner
  model
  options

  function AlwaysSameLabelLearner(options=Dict())
    default_options = {
      :output => :class,
      :label => nothing
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(awsl::AlwaysSameLabelLearner, instances::Matrix, labels::Vector)
  if awsl.options[:label] == nothing
    awsl.model = {
      :label => first(labels)
    }
  else
    awsl.model = {
      :label => awsl.options[:label]
    }
  end
end

function transform!(awsl::AlwaysSameLabelLearner, instances::Matrix)
  return fill(awsl.model[:label], size(instances, 1))
end

end # module
