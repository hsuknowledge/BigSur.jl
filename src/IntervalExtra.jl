import SpecialFunctions.erfc as erfc
import ForwardDiff: Dual, value, partials

_string(x) = string(x)
_string(x::Dual) = string(value(x)) * "+" * string(partials(x))

# Probabilist's Hermite polynomials
const He1 = IntervalPolynomial((0, 1, 0, 0, 0, 0))
const He2 = IntervalPolynomial((-1, 0, 1, 0, 0, 0))
const He3 = IntervalPolynomial((0, -3, 0, 1, 0, 0))
const He4 = IntervalPolynomial((3, 0, -6, 0, 1, 0))
const He5 = IntervalPolynomial((0, 15, 0, -10, 0, 1))
# Cornish-Fisher expansion polynomials
const h1, h2,  h11  = He2/6, He3/24, -(2He3 + He1)/36
const h3, h12, h111 = He4/120, -(He4 + He2)/24, (12He4 + 19He2)/324
const h4, h22, h13  = He5/720, -(3He5 + 6He3 + 2He1)/384, -(2He5 + 3He3)/180
const h112, h1111 = (14He5 + 37He3 + 8He1)/288, -(252He5 + 832He3 + 227He1)/7776

## From https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/113
erfc(a::Interval) = interval(prevfloat(erfc(sup(a))), nextfloat(erfc(inf(a))))
## From StatsFuns.jl (dispatched method from Distributions.ccdf(Normal(), z))
##   normccdf(z::Number) = erfc(z * invsqrt2) / 2
## And invsqrt2 is calculated from a BigFloat:
##   IrrationalConstants::Invsqrt2 = inv(sqrt(big(2)))
const Invsqrt2 = inv(sqrt(big(2)))
normccdf(z::Interval{T}) where {T} = erfc(z * interval(T, Invsqrt2)) / 2
