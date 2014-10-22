# Structures
module Structures

export OCDM,
       Variable,
       NominalVar,
       OrdinalVar,
       NumericVar,
       CDM

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
type CDM
  mat::AbstractMatrix{Float64}
  ctx::Dict{Symbol, Any}

  CDM(mat::AbstractMatrix{Float64}) = new(mat, Dict{Symbol, Any}())
  CDM(mat::AbstractMatrix{Float64}, ctx::Dict{Symbol, Any}) = new(mat, ctx)
end
Base.endof(x::CDM) = endof(x.mat)
Base.ndims(x::CDM) = ndims(x.mat)
Base.size(x::CDM) = size(x.mat)
Base.size(x::CDM, d) = size(x.mat, d)
Base.isequal(x::CDM, y::CDM) = isequal(x.mat, y.mat) && isequal(x.ctx, y.ctx)
Base.getindex(x::CDM, inds...) = getindex(x.mat, inds...)
Base.getindex(x::CDM, key::Symbol) = getindex(x.ctx, key)
Base.setindex!(x::CDM, val, inds...) = setindex!(x.mat, val, inds...)
Base.setindex!(x::CDM, val, key::Symbol) = setindex!(x.ctx, val, key)
Base.deepcopy(x::CDM) = CDM(deepcopy(x.mat), deepcopy(x.ctx))

# Orchestra context-driven matrix.
typealias OCDM CDM

end # module
