# export ou_transition_matrix

function ou_transition_matrix(n)
    Mⱼₖ = zeros(n + 1, n + 1)
    δ(j, k) = (j == k) ? 1 : 0

    for j in 0:n, k in 0:n
        jj = j + 1
        kk = k + 1
        Mⱼₖ[jj, kk] =
            (-n * δ(j, k) + k * δ(j + 1, k) + (n - k) * δ(j - 1, k)) / 2
    end
    return Mⱼₖ
end
