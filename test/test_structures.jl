# Structures tests.
module TestStructures

using FactCheck
using Orchestra.Structures

facts("Orchestra data structs") do
  context("CDM handles indexing getters/setters") do
    cdm = CDM(eye(3))
    cdm[:a] = :b
    cdm[1, 1, 1] = 3

    @fact cdm[:] == cdm.mat[:] => true
    @fact cdm[:, :] == cdm.mat[:, :] => true
    @fact cdm[:a] == cdm.ctx[:a] == :b => true
    @fact cdm[1, 1, 1] == cdm.mat[1, 1, 1] == 3 => true
  end

  context("CDM provides size") do
    cdm = CDM(eye(3))

    @fact size(cdm) == size(cdm.mat) == (3, 3) => true
    @fact size(cdm, 1) == size(cdm.mat, 1) == 3 => true
  end

  context("CDM deep copies") do
    cdm = CDM(eye(3))
    cdm[:a] = [1, 2]
    new_cdm = deepcopy(cdm)
    push!(new_cdm[:a], 3)

    @fact new_cdm[:a] != cdm[:a] => true
  end

  context("CDM tests equality") do
    x_cdm = CDM(eye(3))
    x_cdm[:a] = :b
    y_cdm = CDM(eye(3))
    y_cdm[:a] = :b
    z_cdm = CDM(eye(3))
    z_cdm[:a] = :c

    @fact isequal(x_cdm, x_cdm) => true
    @fact isequal(x_cdm, y_cdm) => true
    @fact isequal(x_cdm, z_cdm) => false
  end
end

end # module
