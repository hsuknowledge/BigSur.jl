function calculator_expected_counts(mat::AbstractMatrix{T}) where {T<:Number}
    gene_totals = sum(mat, dims = 2) |> vec
    cell_totals = sum(mat, dims = 1) |> vec
    @assert all(gene_totals .> 0) "Some genes have zero counts. Please remove them."
    @assert all(cell_totals .> 0) "Some cells have zero counts. Please remove them."
    total = sum(cell_totals)
    @inline function (i, j, alt_gene_total::Integer = 0)
        @boundscheck checkbounds(mat, i, j)
        alt_gene_total != 0 && return alt_gene_total * cell_totals[j] / total
        gene_totals[i] * cell_totals[j] / total
    end
end

function calculator_coefficient_variation(calc_mu)
    mat = calc_mu.mat
    m, n = size(mat)
    gmean = calc_mu.gene_totals ./ n
    model = DataFrame(gene_mean = gmean, mcFano = zero(gmean), mask = trues(m))
    c = zeros(eltype(gmean), 1)
    function (c_search = nothing, rowmask::Union{BitArray, Nothing} = nothing)
        if !isnothing(c_search)
            c[1] = c_search
            if isnothing(rowmask) rowmask = trues(m) else model.mask .= rowmask end
            @tasks for i in findall(identity, rowmask)
                model.mcFano[i] = mapreduce(+, 1:n) do j
                    pearson_residual(mat[i, j], calc_mu(i, j), c_search)^2
                end / (n - 1)
            end
        end
        lm(@formula(log10(mcFano) ~ log10(gene_mean)), model[model.mask, :])
    end
end

function find_coefficient_variation!(calc_cv, rowmask)
    result = optimize(c -> abs(coef(calc_cv(c, rowmask))[2]), 0, 1)
    c = result.minimizer
    calc_cv(c)
    (; c = c, best_slope = result.minimum)
end

function calculator_nulldistribution_Fano(calc_cv)
    S2 = stirlings2_table(12)
    function (i)
        n, c = calc_cv.n, calc_cv.c[1]
        k = @tasks for j in 1:n
            @set reducer = +
            @local tmp = zeros(13+6+6)
            @views r, e, q = tmp[1:13], tmp[14:19], tmp[20:25]
            map!(k -> poisson_lognormal_moment(k, m, c, S2), r, 0:12)
            map!(k -> pearson_residual_moment(2k, m, c, r), e, 1:6)
            noncentral_moment_to_cumulant!(e, q)
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
        quantile_Cornish_Fisher(μ, sd, γ1, γ2, γ3, γ4)
        # pval = ccdf(Normal(), min_abs_real(roots(f(i) - mcFano[i])))
    end
end

function calculator_modifiedcorrected_PCC(calc_cv)
    # inner product of rows i1 and i2 of residuals divided by respective fanos
    function (i1, i2)
        f = calc_cv
        mat, n, c, mcFano, calc_mu = f.mat, f.n, f.c[1], f.model.mcFano, f.calc_mu
        tmapreduce(+, 1:n) do j
            mapreduce(*, [i1, i2]) do i
                pearson_residual(mat[i, j], calc_mu(i, j), c) / sqrt(mcFano[i])
            end
        end / (n - 1)
    end
end

@inline function simulation_gene_levels(gene_totals, n)
    a = max(2, minimum(gene_totals))
    e = n / 50
    h = maximum(gene_totals)
    @. Int(round([ # ordered from least to most, 9 points
        a, a^(3/4) * e^(1/4), sqrt(a * e), a^(1/4) * e^(3/4),
        e, e^(3/4) * h^(1/4), sqrt(e * h), e^(1/4) * h^(3/4), h
    ]))
end

@inline function simulation_trials(sim_total, n)
    a = log10(sim_total)
    Int(round(4e7 / n / (a^(1/5) + 0.5a^3)))
end

@inline function poissonLogNormal_sample(m, c)
    chi = 1 + c^2
    rate = rand(LogNormal(log(m / sqrt(chi)), sqrt(log(chi))))
    rand(Poisson(rate))
end
