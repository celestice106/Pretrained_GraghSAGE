"""
This module generates the temporal event chains that form the foundational structure
of Memory-R1 graphs. These chains represent sequential memories (e.g., a user's
session of interactions).

Key Concepts:
- MemoryEntry Nodes: Represent discrete events/facts
- NEXT_EVENT Edges: Create temporal sequences (E1 → E2 → E3)
- Sessions: Independent chains representing different episodes

Output: Disconnected directed acyclic graphs (DAGs) that will be connected
        by entity hubs in Phase 2.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BackboneConfig:
    """
    Attributes:
        num_sessions: Number of independent event chains to generate
        avg_chain_length: Mean length of each session (Poisson parameter λ)
        chain_length_std: Standard deviation for chain length variation
        min_chain_length: Minimum events per session (avoid degenerate chains)
        max_chain_length: Maximum events per session (avoid unrealistic chains)
    """
    num_sessions: int = 50
    avg_chain_length: float = 10.0
    chain_length_std: float = 3.0
    min_chain_length: int = 3
    max_chain_length: int = 50


class TemporalBackboneGenerator:
    """
    The generated structure consists of:
    - Multiple independent chains (sessions)
    - Directed edges representing temporal progression
    - Timestamps for each event
    - Node metadata (session_id, position_in_session)
    
    Example:
        Session 1: [E1] → [E2] → [E3] → [E4]
        Session 2: [E5] → [E6]
        Session 3: [E7] → [E8] → [E9]
    """
    
    def __init__(self, config: BackboneConfig = None):
        """
        Initialize the backbone generator.
        
        Args:
            config: Configuration parameters. 
        """
        self.config = config or BackboneConfig()
        self.node_counter = 0  # Global node ID counter
        
    def generate(self, seed: int = None) -> nx.DiGraph:
        """
        Generate a temporal backbone graph with multiple session chains.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            NetworkX DiGraph with MemoryEntry nodes and NEXT_EVENT edges
            
        Graph Properties:
            - Nodes have attributes: 'node_type', 'session_id', 'position', 'timestamp'
            - Edges have attribute: 'edge_type' = 'NEXT_EVENT'
            - Multiple disconnected components (one per session)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize empty directed graph
        G = nx.DiGraph()
        
        # Track metadata for later analysis
        metadata = {
            'num_sessions': self.config.num_sessions,
            'total_nodes': 0,
            'total_edges': 0,
            'chain_lengths': []
        }
        
        # Generate each session independently
        for session_id in range(self.config.num_sessions):
            chain_length = self._sample_chain_length()
            session_nodes = self._create_session_chain(
                G, 
                session_id=session_id,
                chain_length=chain_length
            )
            
            metadata['chain_lengths'].append(chain_length)
            metadata['total_nodes'] += len(session_nodes)
            metadata['total_edges'] += len(session_nodes) - 1  # n nodes = n-1 edges
            
        # Store metadata in graph
        G.graph['metadata'] = metadata
        G.graph['generation_phase'] = 1  # Phase 1 complete
        
        return G
    
    def _sample_chain_length(self) -> int:
        """
        Create realistic variation in session lengths:
        - Most sessions are near the average
        - Some sessions are shorter/longer
        - No degenerate (too short) or unrealistic (too long) chains
        
        Returns:
            Integer chain length in [min_chain_length, max_chain_length]
        """
        # Sample from normal distribution
        length = np.random.normal(
            loc=self.config.avg_chain_length,
            scale=self.config.chain_length_std
        )
        
        # Clip to valid range
        length = int(np.clip(
            length,
            self.config.min_chain_length,
            self.config.max_chain_length
        ))
        
        return length
    
    def _create_session_chain(
        self,
        G: nx.DiGraph,
        session_id: int,
        chain_length: int
    ) -> List[str]:
        """
        Create a single temporal chain (session) in the graph.
        
        Args:
            G: Graph to add nodes/edges to (modified in-place)
            session_id: Unique identifier for this session
            chain_length: Number of events in this session
            
        Returns:
            List of node IDs created in this session
            
        Creates:
            - chain_length MemoryEntry nodes
            - chain_length - 1 NEXT_EVENT edges
            - Timestamps with realistic time gaps
        """
        session_nodes = []
        
        # Generate base timestamp for this session (random day in 2024)
        base_timestamp = self._generate_base_timestamp()
        
        # Create nodes and edges in sequence
        for position in range(chain_length):
            # Generate unique node ID
            node_id = f"mem_{self.node_counter}"
            self.node_counter += 1
            
            # Calculate timestamp (events spaced by 1-60 minutes)
            time_offset = timedelta(
                minutes=np.random.randint(1, 60) * position
            )
            timestamp = base_timestamp + time_offset
            
            # Add node with full metadata
            G.add_node(
                node_id,
                node_type='MemoryEntry',  # Schema-required type
                session_id=session_id,
                position=position,  # Position within session (0 = start)
                timestamp=timestamp.isoformat(),
                # Structural metadata (used for validation later)
                is_session_start=(position == 0),
                is_session_end=(position == chain_length - 1)
            )
            
            session_nodes.append(node_id)
            
            # Add edge to previous node (if not first node)
            if position > 0:
                prev_node = session_nodes[position - 1]
                G.add_edge(
                    prev_node,
                    node_id,
                    edge_type='NEXT_EVENT',  # Schema-required type
                    weight=1.0  # Can be used for temporal distance weighting
                )
        
        return session_nodes
    
    def _generate_base_timestamp(self) -> datetime:
        """
        Generate a random base timestamp for a session.
        
        Returns:
            Random datetime in 2024
        """
        # Random day in 2024
        day_of_year = np.random.randint(1, 366)
        base_date = datetime(2024, 1, 1) + timedelta(days=day_of_year)
        
        # Random hour of day
        hour = np.random.randint(0, 24)
        base_date = base_date.replace(hour=hour)
        
        return base_date
    
    def get_statistics(self, G: nx.DiGraph) -> Dict:
        """   
        Args:
            G: Generated graph
            
        Returns:
            Dictionary with structural statistics
        """
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_sessions': self.config.num_sessions,
            'avg_chain_length': np.mean(G.graph['metadata']['chain_lengths']),
            'std_chain_length': np.std(G.graph['metadata']['chain_lengths']),
            'min_chain_length': min(G.graph['metadata']['chain_lengths']),
            'max_chain_length': max(G.graph['metadata']['chain_lengths']),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'num_connected_components': nx.number_weakly_connected_components(G)
        }
        
        # Degree distribution (should be mostly 1-2 for chains)
        degrees = [G.degree(node) for node in G.nodes()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        
        return stats


def visualize_backbone(G: nx.DiGraph, max_nodes: int = 50):
    """
    Args:
        G: Generated backbone graph
        max_nodes: Maximum nodes to display (avoid clutter)
    """
    import matplotlib.pyplot as plt
    
    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        sampled_nodes = list(G.nodes())[:max_nodes]
        G_vis = G.subgraph(sampled_nodes)
    else:
        G_vis = G
    
    # Color nodes by session
    node_colors = []
    for node in G_vis.nodes():
        session_id = G_vis.nodes[node]['session_id']
        node_colors.append(session_id)
    
    # Layout: hierarchical for temporal chains
    pos = nx.spring_layout(G_vis, k=2, iterations=50)
    
    plt.figure(figsize=(12, 8))
    nx.draw(
        G_vis,
        pos,
        node_color=node_colors,
        node_size=300,
        cmap='tab10',
        with_labels=True,
        font_size=8,
        arrows=True,
        arrowsize=10,
        edge_color='gray',
        alpha=0.7
    )
    plt.title(f"Temporal Backbone (Phase 1) - Showing {G_vis.number_of_nodes()} nodes")
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL BACKBONE GENERATOR - PHASE 1 DEMO")
    print("=" * 70)
    print()
    
    # Example 1: Basic Generation
    print("Example 1: Basic Backbone Generation")
    print("-" * 70)
    
    config = BackboneConfig(
        num_sessions=10,
        avg_chain_length=8,
        chain_length_std=2,
        min_chain_length=3,
        max_chain_length=15
    )
    
    generator = TemporalBackboneGenerator(config)
    G = generator.generate(seed=42)
    
    print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print()
    
    # Example 2: Inspect Structure
    print("Example 2: Graph Structure Analysis")
    print("-" * 70)
    
    stats = generator.get_statistics(G)
    print(f"Number of sessions: {stats['num_sessions']}")
    print(f"Average chain length: {stats['avg_chain_length']:.2f} ± {stats['std_chain_length']:.2f}")
    print(f"Chain length range: [{stats['min_chain_length']}, {stats['max_chain_length']}]")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Is DAG: {stats['is_dag']}")
    print(f"Connected components: {stats['num_connected_components']} (should equal num_sessions)")
    print()
    
    # Example 3: Inspect Individual Nodes
    print("Example 3: Sample Node Attributes")
    print("-" * 70)
    
    sample_nodes = list(G.nodes())[:3]
    for node in sample_nodes:
        attrs = G.nodes[node]
        print(f"Node: {node}")
        print(f"  Type: {attrs['node_type']}")
        print(f"  Session: {attrs['session_id']}")
        print(f"  Position: {attrs['position']}")
        print(f"  Timestamp: {attrs['timestamp']}")
        print(f"  Is start: {attrs['is_session_start']}")
        print(f"  Is end: {attrs['is_session_end']}")
        print()
    
    # Example 4: Inspect Edges
    print("Example 4: Sample Edge Attributes")
    print("-" * 70)
    
    sample_edges = list(G.edges())[:3]
    for u, v in sample_edges:
        edge_data = G.edges[u, v]
        print(f"Edge: {u} → {v}")
        print(f"  Type: {edge_data['edge_type']}")
        print(f"  Weight: {edge_data['weight']}")
        print()
    
    # Example 5: Session Analysis
    print("Example 5: Session-by-Session Breakdown")
    print("-" * 70)
    
    for session_id in range(min(3, config.num_sessions)):  # Show first 3 sessions
        session_nodes = [
            node for node in G.nodes()
            if G.nodes[node]['session_id'] == session_id
        ]
        print(f"Session {session_id}: {len(session_nodes)} events")
        print(f"  Start node: {session_nodes[0]}")
        print(f"  End node: {session_nodes[-1]}")
        print()
    
    # Example 6: Degree Distribution Validation
    print("Example 6: Degree Distribution (Should be mostly 1-2)")
    print("-" * 70)
    
    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    
    print(f"In-degree distribution:")
    print(f"  0 (session starts): {in_degrees.count(0)}")
    print(f"  1 (middle nodes): {in_degrees.count(1)}")
    print(f"  2+ (should be 0): {sum(1 for d in in_degrees if d >= 2)}")
    print()
    print(f"Out-degree distribution:")
    print(f"  0 (session ends): {out_degrees.count(0)}")
    print(f"  1 (middle nodes): {out_degrees.count(1)}")
    print(f"  2+ (should be 0): {sum(1 for d in out_degrees if d >= 2)}")
    print()
    
    # Example 7: Save Graph
    print("Example 7: Saving Graph")
    print("-" * 70)
    print("Graph can be saved using:")
    print("  nx.write_gpickle(G, 'backbone.gpickle')")
    print("  nx.write_graphml(G, 'backbone.graphml')")
    print()
    
    print("=" * 70)
    print("Phase 1 complete! Next: Phase 2 (Entity Injection)")
    print("=" * 70)
    
    # Uncomment to visualize (requires matplotlib)
    # visualize_backbone(G, max_nodes=30)