clear; clc;
addpath(genpath('../../../../matlab/'));

%% Change this to your point cloud data path
vel_dir = '/media/zyc/94E4855FE4854508/kitti/sequences/00/velodyne/';
%% use rangenet++ and Euclidean clustring algorithm to extract centroids of semantics
centroid_path = 'data/00/centroids/';
%% a file that record the number of each kinds of semantics.
%% the order in this example file is: vehicle trunk pole
semantic_vector_database = load('data/00/bow.txt');
writematrix(semantic_vector_database, "results/semantic_database.csv");

for idx_query = 1: length(semantic_vector_database)
    %% pick out the semantic vector and centroids information of the query point cloud.
    semantic_vector_query = semantic_vector_database(idx_query,:);
    query_centroids = loadCentroids(centroid_path, idx_query-1);
    
    writematrix(query_centroids, sprintf("results/query_centroids_%d.csv", idx_query));
    
    %% establish graph descriptor and vertices descriptor respectively.
    query_graphdes = Centroids2GraphDes(query_centroids,semantic_vector_query);
    query_verticesdes = GetVertexDes(query_centroids,semantic_vector_query);
    
    writematrix(query_graphdes, sprintf("results/query_graph_desc_%d.csv", idx_query));
    writematrix(query_verticesdes, sprintf("results/query_vertex_desc_%d.csv", idx_query));
end