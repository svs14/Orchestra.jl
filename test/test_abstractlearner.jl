module TestAbstractLearner

using FactCheck
using Fixtures

using Orchestra.AbstractLearner

include(joinpath("learners", "fixture_learners.jl"))
using .FixtureLearners

# Context setup
@fixture function fix_rand()
  srand(1)
  yield_fixture()
end
add_fixture(:context, fix_rand)

facts("Orchestra Learner functions", using_fixtures) do
  context("score calculates accuracy", using_fixtures) do
    learner = StubLearner()
    @fact score(learner, stub_instances, stub_labels, stub_predictions) => 75.0
  end
  context("score throws exception on unknown metric", using_fixtures) do
    learner = StubLearner({:metric => :fake})
    @fact_throws score(learner, stub_instances, stub_labels, stub_predictions)
  end
end

end # module
