@inline var_poisson_lognormal(mu, c) = mu * (1 + c^2 * mu)
@inline pearson_residual(x, mu, c) = (x - mu) / sqrt(var_poisson_lognormal(mu, c))
@inline pearson_residual2(x, mu, c) = (x - mu)^2 / var_poisson_lognormal(mu, c)
stirlings2_table(n) = [stirlings2(k, j) for k in 0:n, j in 0:n]
binomial_table(n) = [binomial(k, i) for k in 0:n, i in 0:n]
@inline function poisson_lognormal_moment(k, m, c, S2)
    acc = 0
    for j in 0:k
        acc += S2[k+1, j+1] * pow(1 + c^2, (j-1)j / 2) * pow(m, j)
    end
    acc
end
@inline function pearson_residual_moment(k, m, c, r, B)
    acc = 0
    for i in 0:k
        acc += (-1)^(k-i) * B[k+1, i+1] * pow(m, k - i) * r[i+1]
    end / pow(var_poisson_lognormal(m, c), k / 2)
end
@inline function pearson_residual2_moment(k, m, c, r, B)
    acc = 0
    for i in 0:2k
        acc += (-1)^i * B[2k+1, i+1] * pow(m, 2k - i) * r[i+1]
    end
    acc / pow(var_poisson_lognormal(m, c), k)
end

@inline function noncentral_moment_to_cumulant!(e, q)
    @assert 4 <= length(e) == length(q) <= 6
    q[1] = e[1]
    q[2] = -e[1]^2 + e[2]
    q[3] = 2e[1]^3 - 3e[2]e[1] + e[3]
    q[4] = -6e[1]^4 + 12e[2]e[1]^2 - 3e[2]^2 - 4e[1]e[3] + e[4]
    length(e) == 4 && return;
    q[5] = 24e[1]^5 - 60e[2]e[1]^3 + 20e[3]e[1]^2 - 10e[2]e[3] +
             30e[2]^2*e[1] - 5e[4]e[1] + e[5]
    length(e) == 5 && return;
    q[6] = -120e[1]^6 + 360e[2]e[1]^4 - 270e[2]^2*e[1]^2 +
             30e[2]^3 - 120e[3]e[1]^3 + 120e[3]e[2]e[1] - 10e[3]^2 +
             30e[4]e[1]^2 - 15e[4]e[2] - 6e[5]e[1] + e[6]
end
@inline function quantile_Cornish_Fisher(μ, σ, γ1, γ2, γ3 = nothing, γ4 = nothing)
    # Probabilist's Hermite polynomials
    He1 = Polynomial([0, 1])
    He2 = Polynomial([-1, 0, 1])
    He3 = Polynomial([0, -3, 0, 1])
    He4 = Polynomial([3, 0, -6, 0, 1])
    He5 = Polynomial([0, 15, 0, -10, 0, 1])
    # Cornish-Fisher expansion polynomials
    h1, h2,  h11  = He2/6, He3/24, -(2He3 + He1)/36
    h3, h12, h111 = He4/120, -(He4 + He2)/24, (12He4 + 19He2)/324
    h4, h22, h13  = He5/720, -(3He5 + 6He3 + 2He1)/384, -(2He5 + 3He3)/180
    h112, h1111 = (14He5 + 37He3 + 8He1)/288, -(252He5 + 832He3 + 227He1)/7776
    # Quantile function: we want to know what quantile an observed value is with
    # respect to the null distribution that has μ, σ, and higher order moments.
    # We are going to find solutions to f(x::quantile(Normal(0, 1), p)) = value,
    # which maps a standard Normal quantile x to the custom distribution.
    # The weight on SD is a polynomial of x, whose first term x == He1.
    w = He1 + mapreduce(*, +, [h1, h2, h11], [γ1, γ2, γ1^2])
    if !isnothing(γ3) w += mapreduce(*, +, [h3, h12, h111], [γ3, γ1*γ2, γ1^3]) end
    if !isnothing(γ4) w += mapreduce(*, +, [h4, h22,  h13,   h112,    h1111],
                                           [γ4, γ2^2, γ1*γ3, γ1^2*γ2, γ1^4]) end
    f = μ + σ * w
end
