# Structures tests.
module TestStructures

using FactCheck
using Orchestra.Structures

facts("Orchestra data structures") do
  context("OCDM handles indexing getters/setters") do
    cdm = OCDM(eye(3))
    cdm[:a] = [:b]
    cdm[1, 1, 1] = 3
    @fact cdm[:] == cdm.mat[:] => true
    @fact cdm[:, :] == cdm.mat[:, :] => true
    @fact cdm[:a] == cdm.ctx[:a] == [:b] => true
    @fact cdm[1, 1, 1] == cdm.mat[1, 1, 1] == 3 => true
  end

  context("OCDM provides size") do
    cdm = OCDM(eye(3))

    @fact size(cdm) == size(cdm.mat) == (3, 3) => true
    @fact size(cdm, 1) == size(cdm.mat, 1) == 3 => true
  end

  context("OCDM deep copies") do
    cdm = OCDM(eye(3))
    cdm[:a] = [1, 2]
    new_cdm = deepcopy(cdm)
    push!(new_cdm[:a], 3)

    @fact new_cdm[:a] != cdm[:a] => true
  end

  context("OCDM tests equality") do
    x_cdm = OCDM(eye(3))
    x_cdm[:a] = [:b]
    y_cdm = OCDM(eye(3))
    y_cdm[:a] = [:b]
    z_cdm = OCDM(eye(3))
    z_cdm[:a] = [:c]

    @fact isequal(x_cdm, x_cdm) => true
    @fact isequal(x_cdm, y_cdm) => true
    @fact isequal(x_cdm, z_cdm) => false
  end

  context("OCDM vcat concatenates") do
    x_cdm = OCDM(eye(3))
    x_cdm[:a] = [1, 2, 3]
    x_cdm[:b] = [:a]
    y_cdm = OCDM(eye(3))
    y_cdm[:a] = [4, 5, 6]
    y_cdm[:b] = [:b]

    e_cdm = OCDM(vcat(eye(3), eye(3)))
    e_cdm[:a] = [1, 2, 3, 4, 5, 6]
    e_cdm[:b] = [:a, :b]
    a_cdm = vcat(x_cdm, y_cdm)
    @fact isequal(a_cdm, e_cdm) => true
  end
  context("OCDM hcat concatenates") do
    x_cdm = OCDM(eye(3))
    x_cdm[:a] = [1, 2, 3]
    x_cdm[:b] = [:a]
    y_cdm = OCDM(eye(3))
    y_cdm[:a] = [1, 2, 3]
    y_cdm[:b] = [:a]

    e_cdm = OCDM(hcat(eye(3), eye(3)))
    e_cdm[:a] = [1, 2, 3]
    e_cdm[:b] = [:a]
    a_cdm = hcat(x_cdm, y_cdm)
    @fact isequal(a_cdm, e_cdm) => true
  end
end

end # module
