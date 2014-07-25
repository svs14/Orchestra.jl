module TestUtil

using FactCheck

importall Orchestra.Types
importall Orchestra.Util

# NOTE(svs14): Decoupled from module FixtureLearners if we do define this.
type StubLearner <: TestLearner
  model
  options

  function StubLearner(options=Dict())
    default_options = {
      :output => :class,
      :label => nothing
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

facts("Orchestra util functions") do
  context("holdout returns proportional partitions") do
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)

    @fact size(left, 1) => n - (n * right_prop)
    @fact size(right, 1) => n * right_prop
    @fact intersect(left, right) => isempty
    @fact size(union(left, right), 1) => n
  end

  context("kfold returns k partitions") do
    num_instances = 10
    num_partitions = 3
    partitions = kfold(num_instances, num_partitions)

    @fact size(partitions, 1) => num_partitions
    [@fact length(partition) >= 6 => true for partition in partitions]
  end

  context("score calculates accuracy") do
    @fact score(
      :accuracy, ["a", "b", "b"], ["a", "b", "b"]
    ) => 100.0
    @fact score(
      :accuracy, ["a", "b", "b"], ["b", "b", "b"]
    ) => not(100.0)
  end

  context("score throws exception on unknown metric") do
    @fact_throws score(
      :fake, ["a"], ["a"]
    )
  end

  context("infer_eltype returns inferred elements type") do
    vector = {1,2,3.0,"a"}
    @fact infer_eltype(vector[1:3]) => Real
  end

  context("infer_eltype returns empty collection's type") do
    vector = Int[]
    @fact infer_eltype(vector) => Int

    vector = []
    @fact infer_eltype(vector) => None
  end

  context("nested_dict_to_tuples produces set of tuples") do
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

  context("nested_dict_set! assigns values") do
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

  context("nested_dict_merge merges two nested dictionaries") do
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
        :d => 5,
        :e => 6
      }
    }
    expected = {
      :a => 4,
      :b => {
        :c => 2,
        :d => 5,
        :e => 6
      }
    }
    actual = nested_dict_merge(first, second)

    @fact actual => expected
  end

  context("create_transformer produces new transformer") do
    learner = StubLearner({:label => :a})
    new_options = {:label => :b}
    new_learner = create_transformer(learner, new_options)

    @fact learner.options[:label] => :a
    @fact new_learner.options[:label] => :b
    @fact true => !isequal(learner, new_learner)
  end
end

end # module
