function calculator_expected_counts(mat::AbstractMatrix{T}) where {T<:Number}
    gene_totals = sum(mat, dims = 2) |> vec
    cell_totals = sum(mat, dims = 1) |> vec
    @assert all(gene_totals .> 0) "Some genes have zero counts. Please remove them."
    @assert all(cell_totals .> 0) "Some cells have zero counts. Please remove them."
    total = sum(cell_totals)
    @inline function (i, j)
        @boundscheck checkbounds(mat, i, j)
        gene_totals[i] * cell_totals[j] / total
    end
end

@inline noise_factor(mu, c) = 1 + c^2 * mu
@inline pearson_residual(x, mu, c) = (x - mu) / sqrt(mu * noise_factor(mu, c))

function calculator_modifiedcorrected_Fano(fun_μij)
    mat = fun_μij.mat
    m, n = size(mat)
    T = typeof(fun_μij.total)
    c = zeros(T, 1)
    cache = zeros(T, m)
    function (c_search = c[1], rows = nothing)
        rows = isnothing(rows) ? (1:m) : rows
        if c[1] != c_search # simply overwrite the cached value
            c[1] = c_search
            @views tmap!(cache[rows], rows) do i
                mapreduce(+, 1:n) do j
                    pearson_residual(mat[i, j], fun_μij(i, j), c_search)^2
                end / (n - 1)
            end
        end
        @views cache[rows]
    end
end

function optimize_c_for_Fano!(fun_mcϕi, sub_rows)
    X = fun_mcϕi.fun_μij.gene_totals[sub_rows] ./ fun_mcϕi.m
    function objective(c)
        Y = fun_mcϕi(c, sub_rows)
        model = lm(@formula(y ~ x), (; x = log10.(X), y = log10.(Y)))
        slope = coef(model)[2]
        abs(slope)
    end
    result = optimize(objective, 0, 1)
    c = result.minimizer
    fun_mcϕi(c) # not returned; instead, fanos_calc.cache is always accessible
    (; c = c, best_slope = result.minimum)
end

function calculator_modifiedcorrected_PCC(fun_mcϕi)
    # inner product of rows i1 and i2 of residuals divided by respective fanos
    function (i1, i2)
        f = fun_mcϕi
        mat, n, c, cache, mu = f.mat, f.n, f.c[1], f.cache, f.fun_μij
        tmapreduce(+, 1:n) do j
            mapreduce(*, [i1, i2]) do i
                pearson_residual(mat[i, j], mu(i, j), c) / sqrt(cache[i])
            end
        end / (n - 1)
    end
end

@inline function stirlings2_table(k_upto)
    table = zeros(Int, k_upto+1, k_upto+1)
    for k in 0:k_upto; for j in 0:k
        table[k+1, j+1] = Combinatorics.stirlings2(k, j)
    end end
    table
end

@inline function poissonLogNormal_moment(k, m, c, stirlings2)
    chi = 1 + c^2
    mapreduce(j -> stirlings2[k+1, j+1] * chi^(j*(j-1)/2) * m^j, +, 0:k)
end

@inline function noncentral_moment_to_cumulant!(e, q)
    @assert 4 <= length(e) == length(q) <= 6
    q[1] = e[1]
    q[2] = -e[1]^2 + e[2]
    q[3] = 2e[1]^3 - 3e[2]e[1] + e[3]
    q[4] = -6e[1]^4 + 12e[2]e[1]^2 - 3e[2]^2 - 4e[1]e[3] + e[4]
    length(e) == 4 && return;
    q[5] = 24e[1]^5 - 60e[2]e[1]^3 + 20e[3]e[1]^2 - 10e[2]e[3] +
             30e[2]^2e[1] - 5e[4]e[1] + e[5]
    length(e) == 5 && return;
    q[6] = -120e[1]^6 + 360e[2]e[1]^4 - 270e[2]^2e[1]^2 +
             30e[2]^3 - 120e[3]e[1]^3 + 120e[3]e[2]e[1] - 10e[3]^2 +
             30e[4]e[1]^2 - 15e[4]e[2] - 6e[5]e[1] + e[6]
end

@inline function unaveraged_nullFano_cumulants(m, c, tmp, stirlings2)
    div = m * noise_factor(m, c) # * (n - 1) for actual moments/cumulants
    @views r, e, q = tmp[1:13], tmp[14:19], tmp[20:25]
    map!(r, 0:12) do k; poissonLogNormal_moment(k, m, c, stirlings2) end
    map!(e, 1:6) do k
        mapreduce(i -> binomial(2k, i) * (-m)^(2k-i) * r[i+1], +, 0:2k) / div^k
    end
    noncentral_moment_to_cumulant!(e, q)
    q
end

function calculator_nulldistribution_Fano(fun_mcϕi)
    S2 = stirlings2_table(12)
    @inline function(i)
        n, c, fn_mu = fun_mcϕi.n, fun_mcϕi.c[1], fun_mcϕi.fun_μij
        k = @tasks for j in 1:n
            @set reducer = +
            @local tmp = zeros(13+6+6)
            unaveraged_nullFano_cumulants(fn_mu(i, j), c, tmp, S2)
        end
        μ = k[1] / (n - 1) # == n/(n-1)
        sk2 = sqrt(k[2])
        sd = sk2 / (n - 1)
        # Skewness, excess kurtosis and so on, over powers of SD
        # The averaging factors (n-1)^k cancel out in these ratios,
        # therefore we skip dividing by them in moments/cumulants.
        γ1 = k[3] / sk2^3
        γ2 = k[4] / sk2^4
        γ3 = k[5] / sk2^5
        γ4 = k[6] / sk2^6
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
        # Quantile function: we want to know what quantile an observed mcFano is,
        # on the null distribution which has n/(n-1) mean and this particular SD,
        # therefore we will find roots to this function minus the observed mcFano.
        # This function maps a standard Normal quantile x to the new distribution.
        # The weight on SD is a polynomial of x, whose first term x == He1.
        f = μ + sd * mapreduce(*, +,
            [He1, h1, h2,  h11, h3,   h12, h111, h4,  h22,   h13,   h112, h1111],
              [1, γ1, γ2, γ1^2, γ3, γ1*γ2, γ1^3, γ4, γ2^2, γ1*γ3, γ2*γ1^2, γ1^4])
        # pval = ccdf(Normal(), min_abs_real(roots(f(i) - mcFano[i])))
    end
end
