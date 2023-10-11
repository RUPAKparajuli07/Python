from collections import deque

def tarjan(g):
    """
    Tarjan's algorithm for finding strongly connected components in a directed graph.

    Args:
    - g: The directed graph represented as an adjacency list.

    Returns:
    - A list of strongly connected components.
    """
    n = len(g)  # Number of vertices in the graph
    stack = deque()  # Stack for DFS traversal
    on_stack = [False for _ in range(n)]  # Tracks whether a node is on the stack
    index_of = [-1 for _ in range(n)]  # Index of each node
    lowlink_of = index_of[:]  # Lowest index reachable from each node

    def strong_connect(v, index, components):
        index_of[v] = index  # Assign the current index to the node
        lowlink_of[v] = index  # Initialize lowlink with the current index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in g[v]:
            if index_of[w] == -1:
                # Recursive call to explore a new node
                index = strong_connect(w, index, components)
                # Update the lowlink for the current node
                lowlink_of[v] = min(lowlink_of[v], lowlink_of[w])
            elif on_stack[w]:
                # Update lowlink if w is on the stack
                lowlink_of[v] = min(lowlink_of[v], index_of[w])

        if lowlink_of[v] == index_of[v]:
            # Found a strongly connected component
            component = []
            w = stack.pop()
            on_stack[w] = False
            component.append(w)
            while w != v:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
            components.append(component)
        return index

    components = []
    for v in range(n):
        if index_of[v] == -1:
            # Start DFS for a new component
            strong_connect(v, 0, components)

    return components

def create_graph(n, edges):
    """
    Create a directed graph represented as an adjacency list.

    Args:
    - n: Number of vertices.
    - edges: List of directed edges as tuples (source, target).

    Returns:
    - The adjacency list representation of the graph.
    """
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
    return g

if __name__ == "__main__":
    # Test
    n_vertices = 7
    source = [0, 0, 1, 2, 3, 3, 4, 4, 6]
    target = [1, 3, 2, 0, 1, 4, 5, 6, 5]
    edges = list(zip(source, target))
    g = create_graph(n_vertices, edges)

    # Find strongly connected components and assert the result
    assert [[5], [6], [4], [3, 2, 1, 0]] == tarjan(g)
