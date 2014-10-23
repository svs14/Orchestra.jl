# Structures
module Structures

export Variable,
       NominalVar,
       OrdinalVar,
       NumericVar,
       CDM,
       OCDM

# Variable.
abstract Variable
# Nominal variable.
immutable NominalVar <: Variable
  levels::Tuple

  NominalVar(levels) = new(tuple(levels...))
end
# Ordinal variable.
immutable OrdinalVar <: Variable
  levels::Tuple

  OrdinalVar(levels) = new(tuple(levels...))
end
# Numeric variable.
immutable NumericVar <: Variable
end
Base.isequal(x::NominalVar, y::NominalVar) = isequal(x.levels, y.levels)
Base.isequal(x::OrdinalVar, y::OrdinalVar) = isequal(x.levels, y.levels)

# Context-driven matrix.
# Should provide access to a matrix and its context (metadata).
# The exact types are left to the implementation.
abstract CDM
Base.endof(x::CDM) = endof(x.mat)
Base.ndims(x::CDM) = ndims(x.mat)
Base.size(x::CDM) = size(x.mat)
Base.size(x::CDM, d) = size(x.mat, d)
Base.isequal(x::CDM, y::CDM) = isequal(x.mat, y.mat) && isequal(x.ctx, y.ctx)
Base.getindex(x::CDM, inds...) = getindex(x.mat, inds...)
Base.getindex(x::CDM, key::Symbol) = getindex(x.ctx, key)
Base.setindex!(x::CDM, val, inds...) = setindex!(x.mat, val, inds...)
Base.setindex!(x::CDM, val, key::Symbol) = setindex!(x.ctx, val, key)
Base.deepcopy{T<:CDM}(x::T) = T(deepcopy(x.mat), deepcopy(x.ctx))

# Orchestra context-driven matrix.
type OCDM <: CDM
  mat::AbstractMatrix{Float64}
  ctx::Dict{Symbol, Vector}

  OCDM(mat::AbstractMatrix{Float64}) = new(mat, Dict{Symbol, Vector}())
  OCDM(mat::AbstractMatrix{Float64}, ctx::Dict{Symbol, Vector}) = new(mat, ctx)
end

# Vertical concatenation of multiple OCDMs.
# First OCDM's context's keys is used
# for context concatenation.
function Base.hcat(xs::OCDM...)
  ocdm_mat = deepcopy(hcat([x.mat for x in xs]...))
  ocdm_ctx = Dict{Symbol, Vector}()
  for key in keys(xs[1].ctx)
    ocdm_ctx[key] = deepcopy(vcat([x[key] for x in xs]...))
  end
  return OCDM(ocdm_mat, ocdm_ctx)
end

# Horizontal concatenation of multiple OCDMS.
# First OCDM's context's keys is used
# as context
function Base.vcat(xs::OCDM...)
  ocdm_mat = deepcopy(vcat([x.mat for x in xs]...))
  ocdm_ctx = deepcopy(xs[1].ctx)
  return OCDM(ocdm_mat, ocdm_ctx)
end

end # module
