import networkx as nx


def region_based_eval(truth, generated):
    G = nx.Graph()
    true_nodes = []
    alg_nodes = []
    for i in range(len(truth)):
        for j in range(len(truth[0])):
            true_seg=str(truth[i][j])+"t"
            alg_seg=str(generated[i][j])+"a"
            if G.has_edge(true_seg, alg_seg) == False:
                G.add_edge(true_seg, alg_seg, weight=1)
            else:
                G[true_seg][alg_seg]['weight'] += 1
            if 'size' not in G.nodes[true_seg]:
                G.nodes[true_seg]['size']=1
                G.nodes[true_seg]['bipartite']=0
                true_nodes.append(true_seg)
            else:
                G.nodes[true_seg]['size']+=1
            if 'size' not in G.nodes[alg_seg]:
                G.nodes[alg_seg]['size']=1
                G.nodes[true_seg]['bipartite']=1
                alg_nodes.append(alg_seg)
            else:
                G.nodes[alg_seg]['size']+=1
    for node in true_nodes:
        for node2 in alg_nodes:
            if G.has_edge(node, node2) == False:
                G.add_edge(node, node2, weight=0)

    matching = nx.bipartite.maximum_matching(G)
    total=0
    for node in alg_nodes:
        match = matching[node]
        intersect = G[match][node]['weight']
        jaccard = intersect/(G.nodes[match]['size']+G.nodes[node]['size']-intersect)
        total+=jaccard
    return total/len(alg_nodes)


true = [[1,1,1,2,2],[1,2,2,2,3],[1,2,4,4,3],[5,5,4,3,3],[5,5,5,5,3]]
generation = [[2,2,1,1,1],[2,2,1,3,3],[2,1,4,3,3],[5,5,4,3,3],[5,5,5,3,3]]


print(region_based_eval(true, generation))
