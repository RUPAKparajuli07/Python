from __future__ import annotations

# Depth-First Search (DFS) to traverse the graph and populate the stack
def dfs(u):
    global graph, reversed_graph, scc, component, visit, stack
    if visit[u]:
        return
    visit[u] = True
    for v in graph[u]:
        dfs(v)
    stack.append(u)

# Second DFS to find strongly connected components and populate 'scc'
def dfs2(u):
    global graph, reversed_graph, scc, component, visit, stack
    if visit[u]:
        return
    visit[u] = True
    component.append(u)
    for v in reversed_graph[u]:
        dfs2(v)

# Kosaraju's Algorithm for finding strongly connected components
def kosaraju():
    global graph, reversed_graph, scc, component, visit, stack
    for i in range(n):
        dfs(i)
    visit = [False] * n
    for i in stack[::-1]:
        if visit[i]:
            continue
        component = []
        dfs2(i)
        scc.append(component)
    return scc

if __name__ == "__main__":
    # Read the number of nodes (n) and edges (m)
    n, m = list(map(int, input().strip().split()))

    graph: list[list[int]] = [[] for _ in range(n)]  # Original graph
    reversed_graph: list[list[int]] = [[] for i in range(n)]  # Reversed graph

    # Input graph data (edges)
    for _ in range(m):
        u, v = list(map(int, input().strip().split()))
        graph[u].append(v)  # Populate the original graph
        reversed_graph[v].append(u)  # Populate the reversed graph

    stack: list[int] = []  # Stack for DFS
    visit: list[bool] = [False] * n  # Keep track of visited nodes
    scc: list[int] = []  # Store the strongly connected components
    component: list[int] = []  # Temporarily store a single component

    # Call Kosaraju's Algorithm to find strongly connected components
    print(kosaraju())
