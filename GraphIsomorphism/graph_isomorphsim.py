import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism


def graph_matching(graph_1, graph_2):
    GM = isomorphism.GraphMatcher(graph_1, graph_2)
    GM.is_isomorphic()
    pi = GM.mapping
    print(GM.mapping)

    plt.figure()
    plt.title('network_G1')
    nx.draw_networkx(G1, node_size=1000, node_color='w', font_size=20)

    plt.figure()
    plt.title('network_G2')
    nx.draw_networkx(G2, node_size=1000, node_color='w', font_size=20)

    return pi


if __name__ == "__main__":
    G1 = nx.path_graph(4)
    G2 = nx.Graph()
    G2.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
    result = graph_matching(G1, G2)

    G1 = nx.Graph()
    G1.add_edges_from([('a', 'g'), ('a', 'h'), ('a', 'i'),
                       ('b', 'g'), ('b', 'h'), ('b', 'j'),
                       ('c', 'g'), ('c', 'i'), ('c', 'j'),
                       ('d', 'h'), ('d', 'i'), ('d', 'j')])
    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (1, 4), (1, 5),
                      (2, 3), (2, 6),
                      (3, 4), (3, 7),
                      (4, 8),
                      (5, 6), (5, 8),
                      (6, 7),
                      (7, 8)])
    result = graph_matching(G2, G1)

    plt.show()
