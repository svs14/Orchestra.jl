module TestUtil

using FactCheck
using Fixtures

using Orchestra.Util

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
    # Check pairwise intersection of partitions
    @fact size([partitions...], 1) => size(unique([partitions...]), 1)
    @fact size(union(partitions...), 1) => num_instances
  end
end

end # module
