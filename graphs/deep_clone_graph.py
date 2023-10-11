from dataclasses import dataclass
from typing import List

@dataclass
class Node:
    value: int = 0
    neighbors: List["Node"] | None = None

    def __post_init__(self) -> None:
        """
        Initializes the neighbors list when a new node is created.
        """
        self.neighbors = self.neighbors or []

    def __hash__(self) -> int:
        """
        Custom hash method to help identify nodes.
        """
        return id(self)

def clone_graph(node: Node | None) -> Node | None:
    """
    Clones a connected undirected graph starting from the given node.
    """
    if not node:
        return None

    originals_to_clones = {}  # Maps original nodes to their clones
    stack = [node]

    # Traverse the graph and create clones of each node
    while stack:
        original = stack.pop()

        if original in originals_to_clones:
            continue

        originals_to_clones[original] = Node(original.value)

        # Add neighbors to the stack for further processing
        stack.extend(original.neighbors or [])

    # Connect cloned nodes to form the cloned graph
    for original, clone in originals_to_clones.items():
        for neighbor in original.neighbors or []:
            cloned_neighbor = originals_to_clones[neighbor]

            if not clone.neighbors:
                clone.neighbors = []

            clone.neighbors.append(cloned_neighbor)

    # Return the clone of the initial node
    return originals_to_clones[node]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
