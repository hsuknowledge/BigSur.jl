#module internal_use
#export IntervalPolynomial, sampled_roots

using Combinatorics: multiset_permutations, with_replacement_combinations
#using IntervalArithmetic
#using IntervalRootFinding

struct IntervalPolynomial{N, T}
    coeffs::NTuple{N, Interval{T}}
end

function IntervalPolynomial(coeffs::NTuple{N, T}) where {N, T}
    IntervalPolynomial(interval.(coeffs))
end

## make it so f is directly callable with x
function (f::IntervalPolynomial)(x)
    mapreduce(i -> f.coeffs[i] * x^(i-1), +, 1:length(f.coeffs))
end

function Base.:+(x::IntervalPolynomial, y::IntervalPolynomial)
    IntervalPolynomial(x.coeffs .+ y.coeffs)
end
function Base.:+(x::Number, y::IntervalPolynomial)
    IntervalPolynomial((interval(x) + y.coeffs[1], y.coeffs[2:end]...))
end
function Base.:-(x::IntervalPolynomial)
    IntervalPolynomial((-).(x.coeffs))
end
function Base.:-(x::IntervalPolynomial, y::Number)
    IntervalPolynomial((x.coeffs[1] - interval(y), x.coeffs[2:end]...))
end
function Base.:*(x::Number, y::IntervalPolynomial)
    IntervalPolynomial(interval(x) .* y.coeffs)
end
function Base.:/(x::IntervalPolynomial, y::Number)
    IntervalPolynomial(x.coeffs ./ interval(y))
end

function as_poly(v::Vector{T}) where {T}
    x -> mapreduce((c, i) -> c * x^(i-1), +, v, 1:length(v))        
end

function sample5point_equidistant(x::Interval{T})::Vector{T} where {T}
    v = mince(x, 4)
    [inf(v[1]), inf(v[2]), mid(x), sup(v[3]), sup(v[4])]
end

## https://stackoverflow.com/questions/77308165/all-permutations-with-replacement
function sampled_roots(f::IntervalPolynomial, region::Interval)
    n = length(f.coeffs)
    coeffs_5points = sample5point_equidistant.(f.coeffs)
    perms = reduce(vcat, collect(multiset_permutations(x, n))
                         for x in with_replacement_combinations(1:5, n))
    polys = (as_poly(getindex.(coeffs_5points, perm)) for perm in perms)
    rts = reduce(vcat, root_region.(rt) for rt in roots.(polys, region))
    #length(rts) == 0 ? interval(0) : hull(rts...; dec = :auto)
end

#end
