import numpy as np
import scipy.misc
from scipy.cluster.vq import vq,kmeans, whiten
from scipy.spatial.distance import euclidean
import pandas as pd


label_to_index = {"SEKER":0,"BARBUNYA":1, "BOMBAY":2,"CALI":3,"HOROZ":4,"SIRA":5,"DERMASON":6}
index_to_label = {label_to_index[key]:key for key in label_to_index}
labeledindex_to_clusterindex = {}
def read_data(path,data_array):
    df = pd.read_excel('DryBeanDataset/Dry_Bean_Dataset.xlsx')
    values = list([sublist[:-1] for sublist in data_array.tolist()])
    tags = list([sublist[-1] for sublist in data_array.tolist()])

    # print(data_array)
    # print(len(data_array))
    # print(values)
    return np.array(values),tags

def use_kmeans(num_clustering, whitened_data):
    center,_ =  kmeans(whitened_data, num_clustering,100)
    cluster = vq(whitened_data, center)
    return center, cluster

def visualize(center, cluster):
    clustersPoints = []
    for i in range(len(cluster)):
        clustersPoints.append([])
    
def estimate(cluster_center,points):
    ps = whiten([point[:-1] for point in points])
    num_points = len(ps)
    
    sums = [euclidean(cluster_center, p) for p in ps]
    #print(sums)
    average_dis = sum(sums)/ float(num_points)
    return average_dis

if __name__ =='__main__':
    df = pd.read_excel('DryBeanDataset/Dry_Bean_Dataset.xlsx')
    data_array =  df.values
    features,tags = read_data('DryBeanDataset/Dry_Bean_Dataset.xlsx',data_array)
    #print(tags)
    rng = np.random.default_rng()
    wf = whiten(features)
    #print(wf)
    centers, cluster = use_kmeans(7, wf)
    all_cluster = list(set(cluster[0]))
    print(all_cluster)
    # relate centers to the labels
    all_data_i = [None]*7
    for i in range(7):
        
        all_data_i[i] = [a for a in data_array.tolist() if label_to_index[a[-1]]==i ]
        #print(len(all_data_i[i]))
    print(centers)
    for i in range(7):
        center = centers[i]
        #print(center)
        average_dis = [None]*7
        for j in range(7):
            average_dis[j] = estimate(center, all_data_i[j])
            print("for clustered p" + str(i) + " to " + str(j) + "th is " + str(average_dis[j]))
        index = average_dis.index(min(average_dis))
        labeledindex_to_clusterindex[index] = i
    



    


    
    


