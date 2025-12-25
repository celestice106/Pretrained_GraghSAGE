"""
This module adds CAUSED_BY edges to create complex causal relationships and
triangular motifs in the graph. These edges represent logical dependencies that
skip across the temporal chain, enabling the model to learn causal reasoning patterns.

Key Concepts:
- CAUSED_BY Edges: Shortcuts connecting causally related events (E1 → E3, skipping E2)
- Triadic Closures: Triangles formed by temporal + causal edges
- Semantic Causality: Events sharing entities are more likely causally related
- Feed-Forward Loops: Pattern E1→E2, E1→E3, E2→E3 (causal triangle)

Output: Complete Memory-R1 synthetic graph with all three structural components:
        Chains (temporal) + Hubs (entities) + Triangles (causality)
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict


@dataclass
class CausalConfig:
    """
    Attributes:
        causal_prob_base: Base probability of creating a causal edge (0.01-0.1)
        causal_boost_shared_entity: Multiplier when events share entity neighbors (2-5x)
        max_causal_distance: Maximum temporal distance for causal links (in hops)
        min_causal_distance: Minimum temporal distance (avoid redundant causality)
        prefer_triangles: If True, boost probability for triangle formation
        triangle_boost: Multiplier for edges that would complete a triangle
    """
    causal_prob_base: float = 0.05
    causal_boost_shared_entity: float = 3.0
    max_causal_distance: int = 10
    min_causal_distance: int = 2
    prefer_triangles: bool = True
    triangle_boost: float = 2.0


class CausalWiring:
    """
    The wiring process:
    1. For each session, identify candidate event pairs (respecting temporal order)
    2. Compute causality probability based on:
       - Base probability
       - Shared entity neighbors (semantic similarity → higher causality)
       - Triangle formation potential
    3. Create CAUSED_BY edges with computed probability
    4. Track resulting motifs (triangles, feed-forward loops)
    """
    
    def __init__(self, config: CausalConfig = None):
        """
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or CausalConfig()
        
    def wire(self, G: nx.DiGraph, seed: int = None) -> nx.DiGraph:
        """
        Add CAUSED_BY edges to create causal relationships.
        
        Args:
            G: Graph from Phase 2 (modified in-place)
            seed: Random seed for reproducibility
            
        Returns:
            Modified graph with causal edges
            
        Graph Changes:
            - Adds CAUSED_BY edges (MemoryEntry → MemoryEntry)
            - Creates triangular motifs
            - Preserves temporal ordering (only earlier → later)
            - Updates graph metadata
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Validate input graph
        if G.graph.get('generation_phase', 0) < 2:
            raise ValueError("Graph must have entity injected")
        
        # Get all sessions
        sessions = self._get_sessions(G)
        
        # Track statistics
        stats = {
            'total_causal_edges': 0,
            'triangles_created': 0,
            'shared_entity_boosts': 0,
            'triangle_boosts': 0
        }
        
        # Wire each session independently
        for session_id, session_nodes in sessions.items():
            session_stats = self._wire_session(G, session_nodes)
            stats['total_causal_edges'] += session_stats['causal_edges']
            stats['triangles_created'] += session_stats['triangles']
            stats['shared_entity_boosts'] += session_stats['entity_boosts']
            stats['triangle_boosts'] += session_stats['triangle_boosts']
        
        # Update graph metadata
        G.graph['generation_phase'] = 3
        G.graph['metadata']['causal_stats'] = stats
        
        return G
    
    def _get_sessions(self, G: nx.DiGraph) -> Dict[int, List[str]]:
        """   
        Args:
            G: Graph to analyze
            
        Returns:
            Dictionary mapping session_id → list of node IDs
        """
        sessions = defaultdict(list)
        
        for node in G.nodes():
            if G.nodes[node].get('node_type') == 'MemoryEntry':
                session_id = G.nodes[node].get('session_id')
                sessions[session_id].append(node)
        
        # Sort each session by position
        for session_id in sessions:
            sessions[session_id].sort(
                key=lambda n: G.nodes[n].get('position', 0)
            )
        
        return sessions
    
    def _wire_session(
        self,
        G: nx.DiGraph,
        session_nodes: List[str]
    ) -> Dict:
        """
        Add causal edges within a single session.
        
        Args:
            G: Graph to modify
            session_nodes: Ordered list of nodes in this session
            
        Returns:
            Statistics about edges created
        """
        stats = {
            'causal_edges': 0,
            'triangles': 0,
            'entity_boosts': 0,
            'triangle_boosts': 0
        }
        
        n = len(session_nodes)
        
        # Consider all pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                node_i = session_nodes[i]
                node_j = session_nodes[j]
                
                # Check temporal distance
                distance = j - i
                if distance < self.config.min_causal_distance:
                    continue
                if distance > self.config.max_causal_distance:
                    break  # No need to check further for this i
                
                # Compute causality probability
                prob = self._compute_causal_probability(
                    G, node_i, node_j, session_nodes
                )
                
                # Track boosts
                if prob > self.config.causal_prob_base:
                    if self._share_entity_neighbors(G, node_i, node_j):
                        stats['entity_boosts'] += 1
                    if self._would_form_triangle(G, node_i, node_j):
                        stats['triangle_boosts'] += 1
                
                # Sample edge creation
                if np.random.random() < prob:
                    G.add_edge(
                        node_i,
                        node_j,
                        edge_type='CAUSED_BY',
                        weight=1.0,
                        temporal_distance=distance
                    )
                    stats['causal_edges'] += 1
                    
                    # Check if this created a triangle
                    if self._forms_triangle(G, node_i, node_j):
                        stats['triangles'] += 1
        
        return stats
    
    def _compute_causal_probability(
        self,
        G: nx.DiGraph,
        node_i: str,
        node_j: str,
        session_nodes: List[str]
    ) -> float:
        """
        Compute probability of creating CAUSED_BY edge from node_i to node_j.
        
        Probability is influenced by:
        1. Base probability (config)
        2. Shared entity neighbors (semantic relatedness)
        3. Triangle formation potential (structural preference)
        
        Args:
            G: Graph
            node_i: Source node (earlier in time)
            node_j: Target node (later in time)
            session_nodes: All nodes in session (for triangle detection)
            
        Returns:
            Probability in [0, 1]
        """
        prob = self.config.causal_prob_base
        
        # Boost 1: Shared entity neighbors
        if self._share_entity_neighbors(G, node_i, node_j):
            prob *= self.config.causal_boost_shared_entity
        
        # Boost 2: Triangle formation
        if self.config.prefer_triangles and self._would_form_triangle(G, node_i, node_j):
            prob *= self.config.triangle_boost
        
        # Clamp to [0, 1]
        prob = min(prob, 1.0)
        
        return prob
    
    def _share_entity_neighbors(
        self,
        G: nx.DiGraph,
        node_i: str,
        node_j: str
    ) -> bool:
        """
        Check if two nodes share at least one entity neighbor.
        
        Args:
            G: Graph
            node_i: First node
            node_j: Second node
            
        Returns:
            True if they share entity neighbors
        """
        # Get entity neighbors (outgoing MENTIONS edges)
        entities_i = set(
            v for v in G.successors(node_i)
            if G.nodes[v].get('node_type') == 'Entity'
        )
        entities_j = set(
            v for v in G.successors(node_j)
            if G.nodes[v].get('node_type') == 'Entity'
        )
        
        return len(entities_i & entities_j) > 0
    
    def _would_form_triangle(
        self,
        G: nx.DiGraph,
        node_i: str,
        node_j: str
    ) -> bool:
        """
        Check if adding edge (i→j) would complete a triangle.
        
        A triangle exists if there's a path i→k→j and we add i→j.
        
        Args:
            G: Graph
            node_i: Source node
            node_j: Target node
            
        Returns:
            True if this edge would complete a triangle
        """
        # Look for intermediate nodes k where i→k and k→j
        successors_i = set(
            v for v in G.successors(node_i)
            if G.nodes[v].get('node_type') == 'MemoryEntry'
        )
        predecessors_j = set(
            v for v in G.predecessors(node_j)
            if G.nodes[v].get('node_type') == 'MemoryEntry'
        )
        
        # If there's overlap, triangle would form
        return len(successors_i & predecessors_j) > 0
    
    def _forms_triangle(
        self,
        G: nx.DiGraph,
        node_i: str,
        node_j: str
    ) -> bool:
        """
        Called after edge is added.
        """
        return self._would_form_triangle(G, node_i, node_j)
    
    def get_statistics(self, G: nx.DiGraph) -> Dict:
        """
        Compute statistics about causal wiring results.
        
        Args:
            G: Graph after causal wiring
            
        Returns:
            Dictionary with causal statistics
        """
        # Get causal edges
        causal_edges = [
            (u, v) for u, v in G.edges()
            if G.edges[u, v].get('edge_type') == 'CAUSED_BY'
        ]
        
        # Temporal distances
        distances = [
            G.edges[u, v].get('temporal_distance', 0)
            for u, v in causal_edges
        ]
        
        # Count triangles (more sophisticated than generation stats)
        triangles = self._count_triangles(G)
        
        # Count feed-forward loops (specific triangle type)
        ffl = self._count_feed_forward_loops(G)
        
        stats = {
            'num_causal_edges': len(causal_edges),
            'avg_temporal_distance': np.mean(distances) if distances else 0,
            'max_temporal_distance': max(distances) if distances else 0,
            'min_temporal_distance': min(distances) if distances else 0,
            'num_triangles': triangles,
            'num_feed_forward_loops': ffl,
            'triangle_density': triangles / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        }
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for u, v in G.edges():
            edge_type = G.edges[u, v].get('edge_type', 'unknown')
            edge_types[edge_type] += 1
        
        stats['edge_type_distribution'] = dict(edge_types)
        
        return stats
    
    def _count_triangles(self, G: nx.DiGraph) -> int:
        """
        Count directed triangles in the graph.
        
        Args:
            G: Graph to analyze
            
        Returns:
            Number of triangles
        """
        triangles = 0
        
        memory_nodes = [
            n for n in G.nodes()
            if G.nodes[n].get('node_type') == 'MemoryEntry'
        ]
        
        # Check all triples
        for i, node_a in enumerate(memory_nodes):
            for node_b in memory_nodes[i+1:]:
                for node_c in memory_nodes[i+2:]:
                    # Check if they form a triangle (any orientation)
                    if (G.has_edge(node_a, node_b) and 
                        G.has_edge(node_b, node_c) and 
                        G.has_edge(node_a, node_c)):
                        triangles += 1
        
        return triangles
    
    def _count_feed_forward_loops(self, G: nx.DiGraph) -> int:
        """
        Feed-forward loop: E1→E2 (NEXT_EVENT), E2→E3 (NEXT_EVENT), E1→E3 (CAUSED_BY)
        
        Args:
            G: Graph to analyze
            
        Returns:
            Number of feed-forward loops
        """
        ffl = 0
        
        memory_nodes = [
            n for n in G.nodes()
            if G.nodes[n].get('node_type') == 'MemoryEntry'
        ]
        
        for node_a in memory_nodes:
            for node_b in memory_nodes:
                if not G.has_edge(node_a, node_b):
                    continue
                if G.edges[node_a, node_b].get('edge_type') != 'NEXT_EVENT':
                    continue
                
                for node_c in memory_nodes:
                    if not G.has_edge(node_b, node_c):
                        continue
                    if G.edges[node_b, node_c].get('edge_type') != 'NEXT_EVENT':
                        continue
                    
                    # Check for causal shortcut
                    if (G.has_edge(node_a, node_c) and 
                        G.edges[node_a, node_c].get('edge_type') == 'CAUSED_BY'):
                        ffl += 1
        
        return ffl
    
    def analyze_motifs(self, G: nx.DiGraph) -> Dict:
        """
        Analyze graph motifs (structural patterns).
        
        Args:
            G: Graph to analyze
            
        Returns:
            Motif statistics
        """
        motifs = {
            'linear_chains': 0,      # a→b (only NEXT_EVENT)
            'causal_shortcuts': 0,   # a→b (CAUSED_BY, skipping temporal chain)
            'star_patterns': 0,      # Multiple events → same entity
            'triangles': 0,          # a→b→c→a
            'feed_forward_loops': 0  # a→b→c with a→c shortcut
        }
        
        # Count motifs
        motifs['triangles'] = self._count_triangles(G)
        motifs['feed_forward_loops'] = self._count_feed_forward_loops(G)
        
        # Count edge types as proxy for other motifs
        for u, v in G.edges():
            edge_type = G.edges[u, v].get('edge_type')
            if edge_type == 'NEXT_EVENT':
                motifs['linear_chains'] += 1
            elif edge_type == 'CAUSED_BY':
                motifs['causal_shortcuts'] += 1
        
        # Count star patterns (entities with high in-degree)
        entity_nodes = [
            n for n in G.nodes()
            if G.nodes[n].get('node_type') == 'Entity'
        ]
        for entity in entity_nodes:
            in_degree = G.in_degree(entity)
            if in_degree >= 3:  # Threshold for "star"
                motifs['star_patterns'] += 1
        
        return motifs


def visualize_with_causality(G: nx.DiGraph, max_nodes: int = 80):
    """
    Visualize complete graph with all three edge types highlighted.
    
    Args:
        G: Complete graph (Phase 3)
        max_nodes: Maximum nodes to display
    """
    import matplotlib.pyplot as plt
    
    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        memory_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'MemoryEntry']
        entity_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity']
        
        sampled_memory = memory_nodes[:int(max_nodes * 0.7)]
        sampled_entity = entity_nodes[:int(max_nodes * 0.3)]
        G_vis = G.subgraph(sampled_memory + sampled_entity)
    else:
        G_vis = G
    
    # Separate node types
    memory_nodes = [n for n in G_vis.nodes() if G_vis.nodes[n].get('node_type') == 'MemoryEntry']
    entity_nodes = [n for n in G_vis.nodes() if G_vis.nodes[n].get('node_type') == 'Entity']
    
    # Layout
    pos = nx.spring_layout(G_vis, k=1.5, iterations=50)
    
    plt.figure(figsize=(16, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G_vis, pos, nodelist=memory_nodes, node_color='skyblue', 
                          node_size=150, label='MemoryEntry', alpha=0.8)
    entity_sizes = [G_vis.nodes[n]['degree'] * 30 + 100 for n in entity_nodes]
    nx.draw_networkx_nodes(G_vis, pos, nodelist=entity_nodes, node_color='salmon',
                          node_size=entity_sizes, label='Entity', alpha=0.8)
    
    # Separate edges by type
    next_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v].get('edge_type') == 'NEXT_EVENT']
    mentions_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v].get('edge_type') == 'MENTIONS']
    causal_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v].get('edge_type') == 'CAUSED_BY']
    
    # Draw edges with different styles
    nx.draw_networkx_edges(G_vis, pos, edgelist=next_edges, edge_color='blue', 
                          alpha=0.3, arrows=True, arrowsize=8, width=1.5, label='NEXT_EVENT')
    nx.draw_networkx_edges(G_vis, pos, edgelist=mentions_edges, edge_color='red', 
                          alpha=0.2, arrows=True, arrowsize=6, width=1.0, label='MENTIONS')
    nx.draw_networkx_edges(G_vis, pos, edgelist=causal_edges, edge_color='green', 
                          alpha=0.5, arrows=True, arrowsize=10, width=2.0, 
                          style='dashed', label='CAUSED_BY')
    
    plt.title("Complete Memory-R1 Graph (Phase 3)\nBlue: Temporal | Red: Entity Mentions | Green: Causality", 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Import previous phases
    from backbone_generator import TemporalBackboneGenerator, BackboneConfig
    from entity_injector import EntityInjector, EntityConfig
    
    print("=" * 70)
    print("CAUSAL WIRING - PHASE 3 DEMO")
    print("=" * 70)
    print()
    
    # Step 1: Generate Phase 1 & 2
    print("Step 1: Generating backbone + entities (Phase 1 & 2)")
    print("-" * 70)
    
    # Phase 1
    backbone_gen = TemporalBackboneGenerator(BackboneConfig(
        num_sessions=15,
        avg_chain_length=12,
        min_chain_length=6,
        max_chain_length=20
    ))
    G = backbone_gen.generate(seed=42)
    print(f"✓ Phase 1: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Phase 2
    injector = EntityInjector(EntityConfig(
        num_entities=40,
        zipf_alpha=1.5,
        mentions_per_event_min=1,
        mentions_per_event_max=4
    ))
    G = injector.inject(G, seed=42)
    print(f"✓ Phase 2: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    
    # Step 2: Add causal edges
    print("Step 2: Adding causal edges (Phase 3)")
    print("-" * 70)
    
    wiring = CausalWiring(CausalConfig(
        causal_prob_base=0.05,
        causal_boost_shared_entity=3.0,
        max_causal_distance=8,
        min_causal_distance=2,
        prefer_triangles=True,
        triangle_boost=2.0
    ))
    G = wiring.wire(G, seed=42)
    print(f"✓ Phase 3: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    
    # Step 3: Analyze results
    print("Step 3: Analyzing causal statistics")
    print("-" * 70)
    
    stats = wiring.get_statistics(G)
    print(f"Causal edges created: {stats['num_causal_edges']}")
    print(f"Average temporal distance: {stats['avg_temporal_distance']:.2f} hops")
    print(f"Temporal distance range: [{stats['min_temporal_distance']}, {stats['max_temporal_distance']}]")
    print(f"Triangles formed: {stats['num_triangles']}")
    print(f"Feed-forward loops: {stats['num_feed_forward_loops']}")
    print(f"Triangle density: {stats['triangle_density']:.4f}")
    print()
    
    print("Edge type distribution:")
    for edge_type, count in stats['edge_type_distribution'].items():
        percentage = count / G.number_of_edges() * 100
        print(f"  {edge_type}: {count} ({percentage:.1f}%)")
    print()
    
    # Step 4: Motif analysis
    print("Step 4: Motif analysis")
    print("-" * 70)
    
    motifs = wiring.analyze_motifs(G)
    print(f"Linear chains (NEXT_EVENT): {motifs['linear_chains']}")
    print(f"Causal shortcuts (CAUSED_BY): {motifs['causal_shortcuts']}")
    print(f"Star patterns (entity hubs): {motifs['star_patterns']}")
    print(f"Triangles: {motifs['triangles']}")
    print(f"Feed-forward loops: {motifs['feed_forward_loops']}")
    print()
    
    # Step 5: Sample causal edges
    print("Step 5: Sample causal edge inspection")
    print("-" * 70)
    
    causal_edges = [
        (u, v) for u, v in G.edges()
        if G.edges[u, v].get('edge_type') == 'CAUSED_BY'
    ]
    
    for u, v in causal_edges[:3]:
        edge_data = G.edges[u, v]
        u_pos = G.nodes[u].get('position', 0)
        v_pos = G.nodes[v].get('position', 0)
        
        print(f"Causal edge: {u} → {v}")
        print(f"  Positions: {u_pos} → {v_pos}")
        print(f"  Temporal distance: {edge_data.get('temporal_distance')} hops")
        print(f"  Forms triangle: {wiring._forms_triangle(G, u, v)}")
        print()
    
    # Step 6: Final graph summary
    print("Step 6: Complete graph summary")
    print("-" * 70)
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"  MemoryEntry: {sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'MemoryEntry')}")
    print(f"  Entity: {sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity')}")
    print()
    print(f"Total edges: {G.number_of_edges()}")
    print(f"  NEXT_EVENT: {stats['edge_type_distribution'].get('NEXT_EVENT', 0)}")
    print(f"  MENTIONS: {stats['edge_type_distribution'].get('MENTIONS', 0)}")
    print(f"  CAUSED_BY: {stats['edge_type_distribution'].get('CAUSED_BY', 0)}")
    print()
    print(f"Graph density: {nx.density(G):.6f}")
    print(f"Is connected: {nx.is_weakly_connected(G)}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print()
    
    print("=" * 70)
    print("🎉 ALL 3 PHASES COMPLETE!")
    print("Graph ready for feature extraction and GraphSAGE training")
    print("=" * 70)
    
    # Uncomment to visualize
    # visualize_with_causality(G, max_nodes=60)