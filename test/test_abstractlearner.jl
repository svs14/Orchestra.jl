module TestAbstractLearner

using FactCheck
using Fixtures

using Orchestra.AbstractLearner

type StubLearner <: Learner
    model
    options

    function StubLearner(options=Dict())
        default_options = {
            :metric => :accuracy
        }
        new(nothing, merge(default_options, options))
    end
end

function train!(stub::StubLearner, instances::Matrix, labels::Vector)
    stub.model = {
        :instances => instances,
        :labels => labels
    }
end
function predict!(stub::StubLearner, instances::Matrix)
    return fill(stub.model[:labels][1], size(instances, 1))
end

stub_instances = [1 1;2 2;3 3; 4 4]
stub_labels = [1;2;3;4]
stub_predictions = [1;2;3;3]

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
