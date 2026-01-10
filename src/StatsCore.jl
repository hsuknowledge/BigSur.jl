@inline var_poisson_lognormal(mu, c) = mu * (1 + abs2(c) * mu)
@inline pearson_residual(x, mu, c) = (x - mu) / sqrt(var_poisson_lognormal(mu, c))
@inline pearson_residual2(x, mu, c) = (x - mu)^2 / var_poisson_lognormal(mu, c)
stirlings2_table(n) = [stirlings2(k, j) for k in 0:n, j in 0:n]
binomial_table(n) = [binomial(k, i) for k in 0:n, i in 0:n]
chipowers_table(n, c) = [(1 + abs2(c))^div(-j + j^2, 2) for j in 0:n]

## note: for some reason unknown, this is faster & far less alloc than mapreduce
@inline function poisson_lognormal_moment(k, m_powers, chi_powers, S2)
    acc = 0
    for j in 0:k
        acc += S2[k+1, j+1] * chi_powers[j+1] * m_powers[j+1]
    end
    acc
end
@inline function pearson_residual_moment(k, m_powers, c, r, B)
    acc = 0
    for i in 0:k
        acc += (-1)^(k-i) * B[k+1, i+1] * m_powers[k-i+1] * r[i+1]
    end
    acc / sqrt(var_poisson_lognormal(m_powers[2], c))^k
end
@inline function pearson_residual2_moment(k, m_powers, c, r, B)
    acc = 0
    for i in 0:2k
        acc += (-1)^i * B[2k+1, i+1] * m_powers[2k-i+1] * r[i+1]
    end
    acc / var_poisson_lognormal(m_powers[2], c)^k
end

@inline noncentral_moment_to_cumulant(e) = [
      e[1],
     -e[1]^2 +   e[2],
     2e[1]^3 -  3e[2]e[1]   +  e[3],
    -6e[1]^4 + 12e[2]e[1]^2 - 4e[3]e[1] - 3e[2]^2 + e[4]
]
# Probabilist's Hermite polynomials
const He1 = IntervalPolynomial((0, 1, 0, 0))
const He2 = IntervalPolynomial((-1, 0, 1, 0))
const He3 = IntervalPolynomial((0, -3, 0, 1))
# Cornish-Fisher expansion polynomials
const h1, h2, h11 = He2/6, He3/24, -(2He3 + He1)/36
"""
Quantile function that translates a standard Normal quantile z to corresponding
quantile of the target distribution. According to Maillard's guide (2012), the
input moments are not the actual moments of the Cornish-Fisher expansion of the
second order. A correction is needed for us to target the resulting parameters.
"""
@inline quantile_Cornish_Fisher(μ, σc, Sc, Kc) =
    μ + σc / sqrt(m2_Cornish_Fisher(Sc, Kc)) * (He1 + Sc*h1 + Kc*h2 + Sc^2*h11)
