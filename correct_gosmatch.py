import numpy as np
from itertools import combinations_with_replacement
from scipy.spatial.distance import pdist, squareform


def build_graph_desc(centroids_vector, semantics_vector):
    desc = []
    semantic_groups = []
    semantic_id = 0
    vec_size = 61

    for sv in semantics_vector:
        semantic_groups.append(centroids_vector[semantic_id:semantic_id + sv])
        semantic_id += sv

    for i, j in list(combinations_with_replacement(range(0, len(semantics_vector)), 2)):
        sgi = semantic_groups[i]
        sgj = semantic_groups[j]

        distances = squareform(pdist(np.concatenate([sgi, sgj])))
        distances = distances[0:len(sgi), len(sgi):]

        bins = [1e-4]
        bins.extend(list(range(1, vec_size)))
        bins.append(1e4)
        hists, bins = np.histogram(distances, bins)

        desc = np.concatenate([desc, hists])

    return desc


def get_vertex_desc(centroids_vector, semantics_vector):
    semantic_id = 0
    vec_size = 61

    distances = squareform(pdist(centroids_vector))
    group_distances = []
    v_des = []

    for sv in semantics_vector:
        group_distances.append(distances[:, semantic_id:semantic_id + sv])
        semantic_id += sv

    for dist in group_distances:
        bins = [1e-4]
        bins.extend(list(range(1, vec_size)))
        bins.append(1e4)
        v_des_per_group = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, dist)
        v_des.append(v_des_per_group)

    v_des = np.hstack(v_des)

    return v_des


def main():
    semantic_database = np.loadtxt("semantic_database.csv", delimiter=",", dtype=np.int64)

    for i in range(len(semantic_database)):
        print("====" * 20, i)
        query = semantic_database[i]
        query_centroids = np.loadtxt(f"results/query_centroids_{i+1}.csv", delimiter=",")
        query_graph_desc_mat = np.loadtxt(f"results/query_graph_desc_{i+1}.csv", delimiter=",")
        query_vertex_desc_mat = np.loadtxt(f"results/query_vertex_desc_{i+1}.csv", delimiter=",")
        query_graph_desc_py = build_graph_desc(query_centroids, query)
        query_vertex_desc_py = get_vertex_desc(query_centroids, query)

        print("====" * 20, "python")
        print(query_graph_desc_py)
        print(query_vertex_desc_py)

        res = query_graph_desc_py == query_graph_desc_mat

        print("====" * 20, "matlab")
        print(query_graph_desc_mat)
        print(query_vertex_desc_mat)

        res = query_vertex_desc_py == query_vertex_desc_mat

        assert res.all()


if __name__ == '__main__':
    main()
