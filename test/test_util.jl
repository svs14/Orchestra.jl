module TestUtil

using FactCheck
using Fixtures

importall Orchestra.Util

include("fixture_learners.jl")
using .FixtureLearners
nfcp = NumericFeatureClassification()

facts("Orchestra util functions", using_fixtures) do
  context("holdout returns proportional partitions", using_fixtures) do
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)

    @fact size(left, 1) => n - (n * right_prop)
    @fact size(right, 1) => n * right_prop
    @fact intersect(left, right) => isempty
    @fact size(union(left, right), 1) => n
  end

  context("kfold returns k partitions", using_fixtures) do
    num_instances = 10
    num_partitions = 3
    partitions = kfold(num_instances, num_partitions)

    @fact size(partitions, 1) => num_partitions
    [@fact length(partition) >= 6 => true for partition in partitions]
  end

  context("score calculates accuracy", using_fixtures) do
    learner = PerfectScoreLearner({:problem => nfcp})
    predictions = fit_and_transform!(learner, nfcp)

    @fact score(
      :accuracy, nfcp.test_labels, predictions
    ) => 100.0
  end
  context("score throws exception on unknown metric", using_fixtures) do
    learner = PerfectScoreLearner({:problem => nfcp})
    predictions = fit_and_transform!(learner, nfcp)

    @fact_throws score(
      :fake, nfcp.test_labels, predictions
    )
  end

  context("infer_eltype returns inferred elements type", using_fixtures) do
    vector = [1,2,3,"a"]
    @fact infer_eltype(vector[1:3]) => Int
  end

  context("nested_dict_to_tuples produces list of tuples", using_fixtures) do
    nested_dict = {
      :a => [1,2],
      :b => {
        :c => [3,4,5]
      }
    }
    expected_set = Set({
      ([:a], [1,2]),
      ([:b,:c], [3,4,5])
    })
    set = nested_dict_to_tuples(nested_dict)

    @fact set => expected_set
  end

  context("nested_dict_set! assigns values", using_fixtures) do
    nested_dict = {
      :a => 1,
      :b => {
        :c => 2
      }
    }
    expected_dict = {
      :a => 1,
      :b => {
        :c => 3
      }
    }
    nested_dict_set!(nested_dict, [:b,:c], 3)

    @fact nested_dict => expected_dict
  end

  context("nested_dict_merge merges two nested dictionaries", using_fixtures) do
    first = {
      :a => 1,
      :b => {
        :c => 2,
        :d => 3
      }
    }
    second = {
      :a => 4,
      :b => {
        :d => 5
      }
    }
    expected = {
      :a => 4,
      :b => {
        :c => 2,
        :d => 5
      }
    }
    actual = nested_dict_merge(first, second)

    @fact actual => expected
  end

  context("create_transformer produces new transformer", using_fixtures) do
    learner = AlwaysSameLabelLearner({:label => :a})
    new_options = {:label => :b}
    new_learner = create_transformer(learner, new_options)

    @fact learner.options[:label] => :a
    @fact new_learner.options[:label] => :b
    @fact true => !isequal(learner, new_learner)
  end
end

end # module
