import numpy as np
from itertools import combinations_with_replacement
from scipy.spatial.distance import pdist, squareform


def build_graph_desc(centroids_vector, semantics_vector):
    # my simplified codes start here
    simple_desc = []
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

        bins = list(range(vec_size))
        bins.append(10000)
        hists, bins = np.histogram(distances, bins)

        simple_desc = np.concatenate([simple_desc, hists])

    # the original codes start here
    desc = []
    tmp = 0

    for i in range(len(semantics_vector)):
        tmp += semantics_vector[i]
        j_tmp = 0

        for j in range(i, len(semantics_vector)):
            j_tmp += semantics_vector[j]

            graph_des_local = cal_local_desc(tmp - semantics_vector[i], tmp, j_tmp - semantics_vector[j], j_tmp, centroids_vector)
            desc = np.concatenate((desc, graph_des_local))

    res = simple_desc == desc

    return desc


def cal_local_desc(i_begin, i_end, j_begin, j_end, centroids_vector):
    vec_size = 61
    interval = 1

    local_desc = np.zeros(vec_size)

    # my simplified codes start here
    centroids_vi = centroids_vector[i_begin:i_end]
    centroids_vj = centroids_vector[j_begin:j_end]

    distances = squareform(pdist(np.concatenate([centroids_vi, centroids_vj])))
    distances = distances[0:len(centroids_vi), len(centroids_vi):]

    bins = list(range(vec_size))
    bins.append(10000)
    hists, bins = np.histogram(distances, bins)

    # the original codes start here
    for i in range(i_begin, i_end):

        for j in range(j_begin, j_end):

            ed = get_distance(centroids_vector[i, :], centroids_vector[j, :])

            for tmp in range(vec_size):

                if tmp * interval <= ed < (tmp + 1) * interval:
                    local_desc[tmp] += 1
                    break

                if ed >= (vec_size - 1) * interval:
                    local_desc[-1] += 1
                    break

    res = hists == local_desc

    return hists


def get_distance(a, b):
    return pdist([a, b], 'euclidean')


def get_vertex_desc(centroids_vector, semantics_vector):
    # my simplified codes start here
    semantic_id = 0
    vec_size = 61

    distances = squareform(pdist(centroids_vector))
    group_distances = []
    sim_v_desc = []

    for sv in semantics_vector:
        group_distances.append(distances[:, semantic_id:semantic_id + sv])
        semantic_id += sv

    for dist in group_distances:
        bins = [1e-4]
        bins.extend(list(range(1, vec_size)))
        bins.append(1e4)
        v_des_per_group = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, dist)
        sim_v_desc.append(v_des_per_group)

    sim_v_desc = np.hstack(sim_v_desc)

    # original codes start here
    index_vector = np.zeros(len(semantics_vector))

    tmp = 0
    interval = 1

    for i in range(len(index_vector)):
        tmp += semantics_vector[i]
        index_vector[i] = tmp  # [10,21,26]

    sem_size = len(semantics_vector)

    v_row_des = np.zeros(vec_size * sem_size)
    v_des = np.zeros((len(centroids_vector), len(v_row_des)))

    for i in range(len(centroids_vector)):

        for j in range(len(centroids_vector)):

            if i == j:
                continue

            ed = get_distance(centroids_vector[i, :], centroids_vector[j, :])

            offset_idx = get_v_des_offset(i, j, index_vector)

            for tmp in range(vec_size):

                if tmp * interval <= ed < (tmp + 1) * interval:
                    v_des[i, tmp + offset_idx * vec_size] += 1
                    break

                if ed >= vec_size * interval:
                    v_des[i, vec_size - 1 + offset_idx * vec_size] += 1
                    break

    res = sim_v_desc == v_des

    return v_des


def get_v_des_offset(i_idx, j_idx, index_vector):
    offset_index = 0

    for iv in index_vector:
        # the original codes are wrong here, to check the consistency with the matlab version, use "j_idx <= iv"
        # to check th consistency with the simplified correct version, use "j_idx < iv"
        if j_idx <= iv:
            break
        else:
            offset_index += 1

    return offset_index


def main():
    semantic_database = np.loadtxt("results/semantic_database.csv", delimiter=",", dtype=np.int64)

    # check the consistency with the original matlab version
    # the loaded results are saved from the matlab side
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
        assert res.all()

        print("====" * 20, "matlab")
        print(query_graph_desc_mat)
        print(query_vertex_desc_mat)

        res = query_vertex_desc_py == query_vertex_desc_mat

        assert res.all()


if __name__ == '__main__':
    main()
