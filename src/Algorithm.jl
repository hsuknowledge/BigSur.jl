function apply_cv!(model::BigSurModel{T}, cv::U; ## cv can be a ForwardDiff.Dual
                   to_all::Bool = false)::U where {T<:Real, U<:Real}
    df = rowdata(model)
    m, n = size(model)
    @assert "for_fitting_cv" in names(df) "Run `set_cv_fitting_range!` first."
    xrows, erows = eachrow(measured_values(model)), eachrow(expected_values(model))
    mask = df.for_fitting_cv
    do_fano = to_all ? trues(m) : mask
    mcFano = typeof(cv) == T ? df.mcFano : zeros(typeof(cv), m)
    @tasks for i in findall(do_fano) ## again, faster and less alloc than mapreduce
        acc = 0
        for (x, mu) in zip(xrows[i], erows[i])
            acc += pearson_residual2(x, mu, cv)
        end
        mcFano[i] = acc / (n - 1)
    end
    prob = CurveFitProblem(df.gene_means[mask], mcFano[mask])
    sol = solve(prob, PowerCurveFitAlgorithm())
    slope = coef(sol)[1] # [slope, intercept] (y=a^x*b)
    @info "search cv = " * _string(cv) * ", slope = " * _string(slope)
    if typeof(cv) == T ## only push if cv is the correct type, avoiding Dual numbers
        df.cv[1] = cv
        push!(cv_searched(model), (cv => slope))
    end
    slope
end

find_cv!(model::BigSurModel{<:Real}) = find_zero(Base.Fix1(apply_cv!, model), 1)
find_cv!(model::BigSurModel{<:Interval}) = begin
    region = interval(0, 1)
    rs = root_region.(roots(Base.Fix1(apply_cv!, model), region))
    while (length(rs) == 0 && sup(region) <= 4) ## we only support search upto <8
        region += sup(region)
        rs = root_region.(roots(Base.Fix1(apply_cv!, model), region))
    end
    length(rs) == 0 && return sup(region) ## return 8; search no further
    length(rs) == 1 && return rs[1]
    hull(rs...; dec = :auto)
end

function quantile_null_mcFano(model::BigSurModel{T}, gene_idx::Integer;
                              order::Integer = 4) where {T<:Real}
    n, c, emat, df = size(model)[2], cv(model), expected_values(model), rowdata(model)
    l = order + 2
    chi_pow = chipowers_table(2l, c)
    S2 = stirlings2_table(2l)
    B = binomial_table(2l)
    k = @tasks for j in 1:n
        @set reducer = +
        @local begin
            m_pow, r = Vector{T}(undef, 2l+1), Vector{T}(undef, 2l+1)
            e, q = Vector{T}(undef, l), Vector{T}(undef, l)
        end
        m_pow .= emat[gene_idx, j].^(0:2l)
        map!(k -> poisson_lognormal_moment(k, m_pow, chi_pow, S2), r, 0:2l)
        map!(k -> pearson_residual2_moment(k, m_pow, c, r, B), e, 1:l)
        noncentral_moment_to_cumulant!(e, q)
        q
    end
    df.null_mu[gene_idx] = μ = k[1] / (n - 1) # == n/(n-1)
    df.null_sd[gene_idx] = sd = sqrt(k[2]) / (n - 1)
    order == 0 && return quantile_Cornish_Fisher(μ, sd)
    # In these ratios, the factors 1/(n-1)^k in the Fano cumulants don't matter.
    γs = @. k[3:l] / sqrt(k[2])^(3:l) # [γ1, γ2, γ3, γ4]
    if l >= 3; df.null_skew[gene_idx] = γs[1] end
    if l >= 4; df.null_ekur[gene_idx] = γs[2]
        df.null_valid[gene_idx] = validity_Cornish_Fisher(γs[1], γs[2])
    end
    quantile_Cornish_Fisher(μ, sd, γs...)
    # pval = ccdf(Normal(), min_abs_real(roots(f(i) - mcFano[i])))
end

function pearson_correlation(model::BigSurModel{T}, i1::Integer, i2::Integer)::T where T
    mcFano, n, c = rowdata(model).mcFano, size(model)[2], cv(model)
    xrow, erow = eachrow(measured_values(model)), eachrow(expected_values(model))
    pres1, pres2 = Vector{T}(undef, n), Vector{T}(undef, n)
    acc = @tasks for j in 1:n
        @set reducer = +
        pres1[j] = pearson_residual(xrow[i1][j], erow[i1][j], c)
        pres2[j] = pearson_residual(xrow[i2][j], erow[i2][j], c)
        pres1[j] * pres2[j]
    end
    acc / (norm(pres1) * norm(pres2))
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
