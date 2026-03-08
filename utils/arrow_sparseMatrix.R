library(arrow)
library(tibble) # for nested data frame
library(Matrix)

#' `as_arrow_table` generic method for Matrix class,
#' stores column-wise compressed (CSC) sparse Matrix and axial metadata.
.S3method("as_arrow_table", "Matrix", function(mat, name = NA) {
  if (!is(mat, "CsparseMatrix")) mat <- as(mat, "CsparseMatrix")
  df <- tibble(
    name = as.character(name),
    rowname = list(rownames(mat)),
    colname = list(colnames(mat)),
    rowidx = list(mat@i),
    colptr = list(mat@p),
    value = list(mat@x)
  )
  field <- arrow::field
  ls_field <- function(t) list_of(field("item", t, nullable = FALSE))
  scm <- schema(c(
    field("name", string()), # this one is nullable, others are not
    purrr::map2(
      c("rowname", "colname", "rowidx", "colptr", "value"),
      c(string(),  string(),  int32(),  int32(),  float64()) |> sapply(ls_field),
      field, nullable = FALSE
    ))
  )
  arrow_table(df, schema = scm)
})

#' Read our specifically formatted arrow file as a sparse Matrix,
#' which consists of a tibble with the following columns:
#' * rowname ==> Matrix@Dimnames[[1]]
#' * colname ==> Matrix@Dimnames[[2]]
#' * rowidx  ==> Matrix@i
#' * colptr  ==> Matrix@p
#' * value   ==> Matrix@x
#' where each row represents a separate sparse Matrix instance.
read_arrow_Matrix <- function(path) {
  tab <- read_ipc_file(path)
  single_mat <- function(i = 1) {
    dimnames <- list(tab$rowname[[i]], tab$colname[[i]])
    m <- new("dgCMatrix")
    m@i <- tab$rowidx[[i]]
    m@p <- tab$colptr[[i]]
    m@x <- tab$value[[i]]
    m@Dim <- sapply(dimnames, length)
    m@Dimnames <- dimnames
    m
  }
  if (nrow(tab) > 1) {
    mats <- lapply(seq_len(nrow(tab)), single_mat)
    if ("name" %in% names(tab)) {
      names(mats) <- tab$name
    }
    mats
  } else {
    single_mat()
  }
}

#' Write a sparse Matrix object or a list of them to an arrow file.
#' This will store these following information for a Matrix (M) as an
#' arrow table with the following columns:
#' * M@Dimnames[[1]] ==> rowname
#' * M@Dimnames[[2]] ==> colname
#' * M@i, M@p, M@x   ==> rowidx, colptr, value, respectively
#' In case of multiple instances, this stores each instance as a row,
#' and each Matrix can have different dimensions.
write_arrow_Matrix <- function(mats, path, name = NA, compression_level = 5) {
  stopifnot(any(c("list", "Matrix") %in% is(mats)))
  if (is(mats, "list")) {
    stopifnot(all(sapply(mats, is, "Matrix")))
    if (is.null(names(mats))) {
      tabs <- lapply(mats, as_arrow_table)
    } else {
      tabs <- mapply(as_arrow_table, mats, names(mats))
    }
    tab <- do.call(rbind, tabs)
  } else { # "Matrix" %in% is(mats)
    tab <- as_arrow_table(mats, name)
  }
  write_ipc_file(tab, path, compression = "zstd", compression_level = compression_level)
}
