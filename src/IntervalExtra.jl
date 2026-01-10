import SpecialFunctions.erfc as erfc
import ForwardDiff: Dual, value, partials

_string(x) = string(x)
_string(x::Dual) = string(value(x)) * "+" * string(partials(x))
_unwrap_number(x) = x
_unwrap_number(x::Interval) = mid(x)
_unwrap_number(x::Dual{T, U}) where {T, U<:Interval} = mid(value(x))

## From https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/113
erfc(a::Interval) = interval(prevfloat(erfc(sup(a))), nextfloat(erfc(inf(a))))
## From StatsFuns.jl (dispatched method from Distributions.ccdf(Normal(), z))
##   normccdf(z::Number) = erfc(z * invsqrt2) / 2
## And invsqrt2 is calculated from a BigFloat:
##   IrrationalConstants::Invsqrt2 = inv(sqrt(big(2)))
const Invsqrt2 = inv(sqrt(big(2)))
normccdf(z::Interval{T}) where {T} = erfc(z * interval(T, Invsqrt2)) / 2
