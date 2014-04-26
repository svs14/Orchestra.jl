module FixtureLearners

importall Orchestra.Types

export MLProblem,
       Classification,
       FeatureClassification,
       NumericFeatureClassification,
       PerfectScoreLearner,
       AlwaysSameLabelLearner,
       train_and_transform!,
       fit!,
       transform!
       
abstract MLProblem
abstract Classification <: MLProblem

# NOTE(svs14): Currently hardcoded example. 
#              Consider turning into rule-based generator.
train_dataset = [
  1.0        1 "b"  2 "c" "a";
  2.0        2 "b"  3 "c" "a";
  nan(3.0)   3 "b"  4 "c" "a";
  -1.0      -1 "d" -2 "c" "b";
  -2.0      -2 "d" -3 "c" "b";
  nan(-3.0) -3 "d" -4 "c" "b";
  1.0        1 "a"  1 "a" "c";
  2.0        2 "b"  2 "b" "c";
  nan(3.0)   3 "c"  3 "c" "c";
  0.0        0 "e"  1 "a" "d";
  0.0        0 "e"  2 "b" "d";
  nan(0.0)   0 "e"  3 "c" "d";
]
test_dataset = [
  4.0        4 "b"  5 "c" "a";
  nan(5.0)   5 "b"  6 "c" "a";
  -4.0      -4 "d" -5 "c" "b";
  nan(-5.0) -5 "d" -6 "c" "b";
  4.0        4 "d"  4 "d" "c";
  nan(5.0)   5 "e"  5 "e" "c";
  0.0        0 "e"  4 "d" "d";
  nan(0.0)   0 "e"  5 "e" "d";
]

type FeatureClassification <: Classification
  train_instances::Matrix
  test_instances::Matrix
  train_labels::Vector
  test_labels::Vector

  function FeatureClassification()
    train_instances = train_dataset[:, 1:end-1]
    test_instances = test_dataset[:, 1:end-1]
    train_labels = train_dataset[:, end]
    test_labels = test_dataset[:, end]
    new(
      train_instances,
      test_instances,
      train_labels,
      test_labels
    ) 
  end
end

type NumericFeatureClassification <: Classification
  train_instances::Matrix
  test_instances::Matrix
  train_labels::Vector
  test_labels::Vector

  function NumericFeatureClassification()
    train_instances = convert(Array{Real, 2}, train_dataset[:, [2,4]])
    test_instances = convert(Array{Real, 2}, test_dataset[:, [2,4]])
    train_labels = convert(Array{String, 1}, train_dataset[:, end])
    test_labels = convert(Array{String, 1}, test_dataset[:, end])
    new(
      train_instances,
      test_instances,
      train_labels,
      test_labels
    ) 
  end
end


function train_and_transform!(learner::Learner, problem::MLProblem, seed=1)
    srand(seed)
    fit!(learner, problem.train_instances, problem.train_labels)
    return transform!(learner, problem.test_instances)
end

type PerfectScoreLearner <: TestLearner
  model
  options

  function PerfectScoreLearner(options=Dict())
    default_options = {
      :output => :class,
      :problem => NumericFeatureClassification()
    }
    new(nothing, merge(default_options, options))
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
    new(nothing, merge(default_options, options))
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
