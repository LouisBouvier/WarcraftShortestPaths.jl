## Graph subtyping

"""
    WarcraftGrid <: AbstractGraph{Int}

Encode the graph corresponding to a Warcraft terrain.

At a given cell position, we can move to any of the 8 cell neighbors in the limits of the grid.
This implies the arcs of the graph, every cell being a node. The cost on each arc is given by 
the cost of the cell destination.

# Fields
- `cell_costs::Matrix{Float16}`: the cell costs labelled in the dataset.
"""
struct WarcraftGrid <: AbstractGraph{Int}
    cell_costs::Matrix{Float16}
end

function node_index(wg::WarcraftGrid, i::Integer, j::Integer)
    n, m = size(wg)
    if (1 <= i <= n) && (1 <= j <= m)
        v = (i - 1) * m + (j - 1) + 1  # enumerate row by row
        return v
    else
        return 0
    end
end

function node_coord(wg::WarcraftGrid, v::Integer)
    n, m = size(wg)
    if 1 <= v <= n * m
        i = (v - 1) ÷ m + 1
        j = (v - 1) % m + 1
        return i, j
    else
        return (0, 0)
    end
end

Base.eltype(::WarcraftGrid) = Int
Graphs.edgetype(::WarcraftGrid) = Graphs.SimpleEdge{Int}

Graphs.is_directed(::WarcraftGrid) = true
Graphs.is_directed(::Type{<:WarcraftGrid}) = true

Base.size(wg::WarcraftGrid, args...) = size(wg.cell_costs, args...)
height(wg::WarcraftGrid) = size(wg, 1)
width(wg::WarcraftGrid) = size(wg, 2)

Graphs.nv(wg::WarcraftGrid) = prod(size(wg))
Graphs.vertices(wg::WarcraftGrid) = 1:nv(wg)
Graphs.has_vertex(wg::WarcraftGrid, v::Integer) = 1 <= v <= nv(wg)

function Graphs.ne(wg::WarcraftGrid)
    n, m = size(wg)
    return (
        max(n - 2, 0) * max(m - 2, 0) * 8 +  # central nodes
        min(m, 2) * max(n - 2, 0) * 5 +  # vertical borders
        min(n, 2) * max(m - 2, 0) * 5 +  # horizontal borders
        min(m, 2) * min(n, 2) * 3  # corners
    )
end

function Graphs.has_edge(wg::WarcraftGrid, s::Integer, d::Integer)
    if has_vertex(wg, s) && has_vertex(wg, d)
        is, js = node_coord(wg, s)
        id, jd = node_coord(wg, d)
        return (s != d) && (abs(is - id) <= 1) && (abs(js - jd) <= 1)  # 8 neighbors max
    else
        return false
    end
end

function Graphs.outneighbors(wg::WarcraftGrid, s::Integer)
    n, m = size(wg)
    i, j = node_coord(wg, s)
    possible_neighbors = (
        (i - 1, j - 1),
        (i - 1, j),
        (i - 1, j + 1),
        (i, j - 1),
        (i, j + 1),
        (i + 1, j - 1),
        (i + 1, j),
        (i + 1, j + 1),
    )  # listed in ascending index order!
    neighbors = (
        node_index(wg, id, jd) for (id, jd) in possible_neighbors if (1 <= id <= n) && (1 <= jd <= m)
    )
    return neighbors
end

Graphs.inneighbors(wg::WarcraftGrid, d::Integer) = outneighbors(wg, d)

function Graphs.edges(wg::WarcraftGrid)
    return (
        Graphs.SimpleEdge(s, d) for s in vertices(wg) for d in outneighbors(wg, s)
    )
end

function Graphs.weights(wg::WarcraftGrid)
    E = edges(wg)
    I = [src(e) for e in E]
    J = [dst(e) for e in E]
    V = Float16[]
    for e in E
        d = dst(e)
        id, jd = node_coord(wg, d)
        cost = wg.cell_costs[id, jd]
        push!(V, cost)
    end
    W = sparse(I, J, V, nv(wg), nv(wg))
    return W
end

# Indexing correspondance bewteen matrix and vector 
function grid_to_vector(label::Matrix{T}) where T
    n, m = size(label)
    vec_label = zeros(T, n*m)
    for i = 1:n
        for j = 1:m
            index = (i - 1) * m + (j - 1) + 1
            vec_label[index] = label[i,j]
        end
    end
    return vec_label
end

function vector_to_grid(vec_label::Vector{T}) where T
    nm = length(vec_label)
    c = Int(sqrt(nm))
    grid = zeros(T, (c, c)) #caution square grid
    for v = 1:nm
        i = (v - 1) ÷ c + 1
        j = (v - 1) % c + 1
        grid[i,j] = vec_label[v]
    end
    return grid
end

## Shortest path function
"""
    warcraft_shortest_path(θ; wg::WarcraftGrid)

Compute the shortest path between the top-left and bottom-right corners of `WarcraftGrid`, given cell costs `θ`.

We use A-star algorithm involving sparse cost representations. Since this function is used in a differentiation 
through argmax pipeline, `θ` is the opposit of the cell costs. We require the latter to be non-negative.
"""
function warcraft_shortest_path(θ; wg::WarcraftGrid)
    Ic = [src(e) for e in edges(wg)]
    Jc = [dst(e) for e in edges(wg)]
    Vc = Float16[]
    for e in edges(wg)
        d = dst(e)
        cost = -θ[d] #stack vector
        push!(Vc, cost)
    end
    c = sparse(Ic, Jc, Vc, nv(wg), nv(wg))

    path = a_star(wg, 1, nv(wg), c)
    y =  zeros(UInt8, nv(wg))
    y[1] = 1 #start
    for edge in path
        d = dst(edge)
        y[d] = 1
    end
    return y
end
