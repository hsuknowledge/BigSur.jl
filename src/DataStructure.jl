struct BigSurModel{T<:Real, A<:AbstractMatrix{T}}
    mat::A
    gene_totals::Vector{T}
    cell_totals::Vector{T}
    rowdata::DataFrame
    cv_searched::Vector{Pair{T, T}}
end

function BigSurModel(mat::AbstractMatrix{T}, names::AbstractVector) where {T<:Real}
    m, n = size(mat)
    @info "Creating analysis model for " * string(size(mat)) * " genes and cells."
    rsum = rowsum(mat)
    csum = colsum(mat)
    @assert all(rsum .> 0) "Some genes have zero counts. Please remove them."
    @assert all(csum .> 0) "Some cells have zero counts. Please remove them."
    rmean = rsum ./ n
    df = DataFrame(names = names, gene_means = rmean,
                   cv = zeros(T, m), mcFano = zeros(T, m),
                   null_mu = zeros(T, m), null_sd = zeros(T, m),
                   null_skew = zeros(T, m), null_ekur = zeros(T, m),
                   null_valid = falses(m))
    BigSurModel(mat, rsum, csum, df, Pair{T, T}[])
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

function expected_values(model::BigSurModel{T},
                         gene_totals::AbstractVector{T} = rowsum(model)
                        ) where {T<:Real}
    csum, total = colsum(model), sum(colsum(model))
    LazyArray(@~ gene_totals * csum' ./ total)
end


function set_cv_fitting_range!(model::BigSurModel, lowbound = 0.1, highbound = 100)
    gmean = rowdata(model).gene_means
    for_fitting_cv = (x -> lowbound < x < highbound).(gmean)
    rowdata(model)[!, :for_fitting_cv] = for_fitting_cv
    num_genes = string(sum(for_fitting_cv))
    @info "Setting " * num_genes * " genes for fitting cv using" lowbound highbound
end
