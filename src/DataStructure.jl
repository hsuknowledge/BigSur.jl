struct BigSurModel{T<:Real, A<:AbstractMatrix{T}}
    mat::A
    gene_totals::Vector{T}
    cell_totals::Vector{T}
    rowdata::DataFrame
    cv_searched::Vector{Pair{T, T}}
    pearson_residual::Matrix{T}
end

function BigSurModel(mat_in::AbstractMatrix{T}, names::AbstractVector;
                     min_genes::Int = 6) where {T<:Real}
    n_gene = rowsum(mat_in .> 0)
    g_filter = n_gene .>= min_genes
    mat = mat_in[g_filter, :]
    m, n = size(mat)
    @info "Creating analysis model for " * string(size(mat)) * " genes and cells."
    rsum = rowsum(mat)
    csum = colsum(mat)
    @assert all(csum .> 0) "Some cells have zero counts. Please remove them."
    rmean = rsum ./ n
    df = DataFrame(names = names[g_filter], gene_means = rmean,
                   fit_cv = falses(m), cv = zeros(T, m), mcFano = zeros(T, m),
                   sum_res = zeros(T, m), k1 = zeros(T, m), k2 = zeros(T, m),
                   k3 = zeros(T, m), k4 = zeros(T, m))
    pres = Matrix{T}(undef, n, m) ## note we switch to genes as columns here
    BigSurModel(mat, rsum, csum, df, Pair{T, T}[], pres)
end

rowsum(mat::AbstractMatrix) = vec(sum(mat, dims = 2))
colsum(mat::AbstractMatrix) = vec(sum(mat, dims = 1))
rowsum(model::BigSurModel) = model.gene_totals
colsum(model::BigSurModel) = model.cell_totals

measured_values(model::BigSurModel) = model.mat
Base.size(model::BigSurModel) = size(model.mat)
rowdata(model::BigSurModel) = model.rowdata
cv(model::BigSurModel) = rowdata(model).cv[1]
cv_searched(model::BigSurModel) = model.cv_searched
pearson_residual(model::BigSurModel) = model.pearson_residual

function expected_values(model::BigSurModel{T},
                         gene_totals::AbstractVector{T} = rowsum(model)
                        ) where {T<:Real}
    csum, total = colsum(model), sum(colsum(model))
    LazyArray(@~ gene_totals * csum' ./ total)
end


function set_cv_fitting_range!(model::BigSurModel, lowbound = 0.1, highbound = 100)
    gmean = rowdata(model).gene_means
    fit_cv = (x -> lowbound < x < highbound).(gmean)
    rowdata(model)[!, :fit_cv] = fit_cv
    num_genes = string(sum(fit_cv))
    @info "Setting " * num_genes * " genes for fitting cv using" lowbound highbound
end
