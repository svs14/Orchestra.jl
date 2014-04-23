module FixtureLearners

importall Orchestra.AbstractLearner

export MLProblem,
       Classification,
       NumericFeatureClassification,
       PerfectScoreLearner,
       AlwaysSameLabelLearner,
       train_and_predict!,
       train!,
       predict!
       
abstract MLProblem
abstract Classification <: MLProblem

type NumericFeatureClassification <: Classification
  train_instances::Matrix
  test_instances::Matrix
  train_labels::Vector
  test_labels::Vector

  function NumericFeatureClassification()
    # NOTE(svs14): Currently hardcoded example. 
    #              Consider turning into rule-based generator.
    train_dataset = [
      1 2 "a";
      2 3 "a";
      3 4 "a";
      -1 -2 "b";
      -2 -3 "b";
      -3 -4 "b";
      1 1 "c";
      2 2 "c";
      3 3 "c";
      0 1 "d";
      0 2 "d";
      0 3 "d";
    ]
    test_dataset = [
      4 5 "a";
      5 6 "a";
      -4 -5 "b";
      -5 -6 "b";
      4 4 "c";
      5 5 "c";
      0 4 "d";
      0 5 "d";
    ]

    train_instances = convert(Array{Real, 2}, train_dataset[:, 1:end-1])
    test_instances = convert(Array{Real, 2}, test_dataset[:, 1:end-1])
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


function train_and_predict!(learner::Learner, problem::MLProblem, seed=1)
    srand(seed)
    train!(learner, problem.train_instances, problem.train_labels)
    return predict!(learner, problem.test_instances)
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

function train!(
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

function predict!(
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

function train!(awsl::AlwaysSameLabelLearner, instances::Matrix, labels::Vector)
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

function predict!(awsl::AlwaysSameLabelLearner, instances::Matrix)
  return fill(awsl.model[:label], size(instances, 1))
end

end # module
