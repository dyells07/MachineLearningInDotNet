namespace Unsupervised

module PCA =

    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Statistics

    // Compute the covariance matrix
    let covarianceMatrix (M: Matrix<float>) =
        let cols = M.ColumnCount
        let mutable C = DenseMatrix.create cols cols 0.0
        for c1 in 0 .. (cols - 1) do
            C.[c1, c1] <- Statistics.Variance(M.Column(c1))
            for c2 in (c1 + 1) .. (cols - 1) do
                let cov = Statistics.Covariance(M.Column(c1), M.Column(c2))
                C.[c1, c2] <- cov
                C.[c2, c1] <- cov
        C

    // Normalize the dataset
    let normalize (dim: int) (observations: float[][]) =
        let averages =
            Array.init dim (fun i ->
                observations |> Array.averageBy (fun x -> x.[i]))

        let stdDevs =
            Array.init dim (fun i ->
                let avg = averages.[i]
                observations
                |> Array.averageBy (fun x -> 
                    pown (x.[i] - avg) 2 |> sqrt))

        observations
        |> Array.map (fun row ->
            row
            |> Array.mapi (fun i x ->
                (x - averages.[i]) / stdDevs.[i]))

    // Perform PCA
    let pca (observations: float[][]) =
        let dataMatrix = Matrix.Build.DenseOfRowArrays(observations)
        let covMatrix = covarianceMatrix dataMatrix
        let factorization = covMatrix.Evd() // Eigenvalue decomposition
        let eigenValues = factorization.EigenValues.Real()
        let eigenVectors = factorization.EigenVectors

        let projector (obs: float[]) =
            let obsVector = Vector.Build.DenseOfArray(obs)
            (eigenVectors.TransposeThisAndMultiply(obsVector))
            |> Vector.toArray

        (eigenValues, eigenVectors), projector
