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

@inline function noncentral_moment_to_cumulant!(e, q)
    @assert 2 <= length(e) == length(q) <= 6
    q[1] = e[1]
    q[2] = -e[1]^2 + e[2]
    length(e) == 2 && return;
    q[3] = 2e[1]^3 - 3e[2]e[1] + e[3]
    length(e) == 3 && return;
    q[4] = -6e[1]^4 + 12e[2]e[1]^2 - 3e[2]^2 - 4e[1]e[3] + e[4]
    length(e) == 4 && return;
    q[5] = 24e[1]^5 - 60e[2]e[1]^3 + 20e[3]e[1]^2 - 10e[2]e[3] +
             30e[2]^2*e[1] - 5e[4]e[1] + e[5]
    length(e) == 5 && return;
    q[6] = -120e[1]^6 + 360e[2]e[1]^4 - 270e[2]^2*e[1]^2 +
             30e[2]^3 - 120e[3]e[1]^3 + 120e[3]e[2]e[1] - 10e[3]^2 +
             30e[4]e[1]^2 - 15e[4]e[2] - 6e[5]e[1] + e[6]
end
@inline function validity_Cornish_Fisher(γ1, γ2)
    criterion1 = abs(γ1) - 6 * (sqrt(2) - 1)
    criterion2 = 27γ2^2 - (216 + 66γ1^2)γ2 + 40γ1^4 + 336γ1^2
    try criterion1 <= 0 && criterion2 <= 0 catch; true end ## catch interval overlap
end
@inline function quantile_Cornish_Fisher(μ, σ, γ1 = nothing, γ2 = nothing,
                                               γ3 = nothing, γ4 = nothing)
    # Quantile function: we want to know what quantile an observed value is with
    # respect to the null distribution that has μ, σ, and higher order moments.
    # We are going to find solutions to f(x::quantile(Normal(0, 1), p)) = value,
    # which maps a standard Normal quantile x to the custom distribution.
    # The weight on SD is a polynomial of x, whose first term x == He1.
    f(w) = μ + σ * w
    w = He1
    isnothing(γ1) && return f(w)
    w += γ1 * h1
    isnothing(γ2) && return f(w)
    w += γ2 * h2 + γ1^2 * h11
    isnothing(γ3) && return f(w)
    w += γ3 * h3 + γ1*γ2* h12 + γ1^3 * h111
    isnothing(γ4) && return f(w)
    w += γ4 * h4 + γ2^2 * h22 + γ1*γ3* h12 + γ1^2 * γ2 * h112 + γ1^4 * h1111
    f = μ + σ * w
end
