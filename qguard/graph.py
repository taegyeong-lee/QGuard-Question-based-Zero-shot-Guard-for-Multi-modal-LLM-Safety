from typing import Iterable, Dict, List, Set
import itertools
import networkx as nx

def build_graph_from_results(
    question_results: Iterable[Dict],
    group_by_category: Dict[str, str],
    group_coupling: float = 0.1,
    intra_group_q_coupling: float = 0.3,
) -> nx.DiGraph:
    G = nx.DiGraph()

    for item in question_results:
        q = item["question"]
        p = float(item["yes_prob"])
        g = group_by_category.get(q, "Unknown")
        G.add_node(q, type="question")
        G.add_node(g, type="group")
        G.add_edge(q, g, weight=max(0.0, min(1.0, p))) 

    groups: Set[str] = {group_by_category.get(it["question"], "Unknown") for it in question_results}
    for g1, g2 in itertools.combinations(groups, 2):
        G.add_edge(g1, g2, weight=group_coupling)

    qs_by_group: Dict[str, List[str]] = {}
    for it in question_results:
        g = group_by_category.get(it["question"], "Unknown")
        qs_by_group.setdefault(g, []).append(it["question"])
    for g, qs in qs_by_group.items():
        for q1, q2 in itertools.combinations(qs, 2):
            G.add_edge(q1, q2, weight=intra_group_q_coupling)

    return G


def pagerank_risk_score(G: nx.DiGraph) -> float:
    if len(G) == 0:
        return 0.0
    pr = nx.pagerank(G, weight="weight")
    score = 0.0
    for n in G.nodes:
        out_w = sum(d.get("weight", 0.0) for _, _, d in G.edges(n, data=True))
        score += pr[n] * out_w
    return float(score)
