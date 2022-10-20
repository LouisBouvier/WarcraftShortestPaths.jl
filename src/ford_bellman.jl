
"""
    grid_bellman_ford_warcraft(g, s, d, length_max)

Apply the Bellman-Ford algorithm on an `GridGraph` `g`, and return a `ShortestPathTree` with source `s` and destination `d`,
among the paths having length smaller than `length_max`.
"""
function grid_bellman_ford_warcraft(g::GridGraph{T,R,W,A}, s::Integer, d::Integer, length_max::Int = nv(g)) where {T,R,W,A}
    # Init storage
    parents = zeros(T, nv(g), length_max+1)
    dists = Matrix{Union{Nothing,R}}(undef, nv(g), length_max+1)
    fill!(dists, Inf)
    # Add source
    dists[s,1] = zero(R)
    # Main loop
    for k in 1:length_max
        for v in vertices(g)
            for u in inneighbors(g, v)
                d_u = dists[u, k]
                if !isinf(d_u)
                    d_v = dists[v, k+1]
                    d_v_through_u = d_u + GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k+1] = d_v_through_u
                        parents[v, k+1] = u
                    end
                end
            end
        end
    end
    # Get length of the shortest path
    k_short = argmin(dists[d,:])
    if isinf(dists[d, k_short])
        println("No shortest path with less than $length_max arcs")
        return T[]
    end
    # Deduce the path
    v = d
    path = [v]
    k = k_short
    while v != s
        v = parents[v, k]
        if v == zero(T)
            return T[]
        else
            pushfirst!(path, v)
            k = k-1
        end
    end
    return path
end