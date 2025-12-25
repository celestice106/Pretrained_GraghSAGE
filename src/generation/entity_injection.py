"""
This module transforms disconnected temporal chains into a connected
graph by adding Entity nodes and MENTIONS edges. Entities act as "hubs" that link
together different memory sequences, mimicking how real-world concepts (people,
places, objects) appear across multiple events.

Key Concepts:
- Entity Nodes: Semantic anchors (e.g., "user", "project_X", "apple")
- MENTIONS Edges: Link MemoryEntry → Entity (event mentions concept)
- Zipfian Distribution: Few entities are very popular (hubs), most are rare
- Heterophily: Connections are functional, not similarity-based

Output: Connected graph where temporal chains are "stapled together" by entity hubs.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class EntityConfig:
    """Configuration for entity hub injection.
    
    Attributes:
        num_entities: Total number of unique entities to create
        zipf_alpha: Power-law exponent for popularity (1.5 = realistic skew)
                    Higher α = more concentrated (few very popular entities)
                    Lower α = more uniform distribution
        mentions_per_event_min: Minimum entities mentioned per event
        mentions_per_event_max: Maximum entities mentioned per event
        ensure_connectivity: If True, ensure all sessions share at least one entity
    """
    num_entities: int = 100
    zipf_alpha: float = 1.5
    mentions_per_event_min: int = 1
    mentions_per_event_max: int = 5
    ensure_connectivity: bool = True


class EntityInjector:
    """
    Injects Entity nodes and MENTIONS edges into temporal backbone graphs.
    
    The injection process:
    1. Creates Entity nodes with popularity weights
    2. For each MemoryEntry, samples entities based on popularity
    3. Creates MENTIONS edges (MemoryEntry → Entity)
    4. Optionally ensures graph connectivity
    """
    
    def __init__(self, config: EntityConfig = None):
        """
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or EntityConfig()
        self.entity_counter = 0
        
    def inject(self, G: nx.DiGraph, seed: int = None) -> nx.DiGraph:
        """
        Args:
            G: Backbone graph from Phase 1 (modified in-place)
            seed: Random seed for reproducibility
            
        Returns:
            Modified graph with entities and mentions edges
            
        Graph Changes:
            - Adds Entity nodes with 'popularity' attribute
            - Adds MENTIONS edges (MemoryEntry → Entity)
            - Transforms disconnected components into connected graph
            - Updates graph metadata
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Validate input graph
        if G.graph.get('generation_phase', 0) < 1:
            raise ValueError("Backbone graph must be generated first!")
        
        # Step 1: Create entity nodes with popularity weights
        entities = self._create_entities(G)
        
        # Step 2: Create MENTIONS edges from MemoryEntry nodes to entities
        self._create_mentions_edges(G, entities)
        
        # Step 3: Optionally ensure connectivity
        if self.config.ensure_connectivity:
            self._ensure_connectivity(G, entities)
        
        # Update metadata
        G.graph['generation_phase'] = 2
        G.graph['metadata']['num_entities'] = len(entities)
        
        return G
    
    def _create_entities(self, G: nx.DiGraph) -> List[Tuple[str, float]]:
        """
        Create Entity nodes with Zipfian popularity distribution.
        
        Args:
            G: Graph to add entities to
            
        Returns:
            List of (entity_id, popularity_weight) tuples, sorted by popularity
        """
        entities = []
        
        # Generate Zipfian weights (power-law distribution)
        ranks = np.arange(1, self.config.num_entities + 1)
        # Weight for rank r:
        #   w = 1/r^alpha
        weights = 1.0 / np.power(ranks, self.config.zipf_alpha)
        weights = weights / np.sum(weights)  # Normalize to probabilities
        
        # Create entity nodes
        for i in range(self.config.num_entities):
            entity_id = f"ent_{self.entity_counter}"
            self.entity_counter += 1
            
            G.add_node(
                entity_id,
                node_type='Entity',  # Schema-required type
                popularity=weights[i],  # Used for sampling
                degree=0,  # Will be updated as edges are added
                entity_rank=i  # Rank in popularity (0 = most popular)
            )
            
            entities.append((entity_id, weights[i]))
        
        return entities
    
    def _create_mentions_edges(
        self,
        G: nx.DiGraph,
        entities: List[Tuple[str, float]]
    ):
        """
        For each MemoryEntry:
        1. Sample k entities (k ~ Uniform[min, max])
        2. Sample entities based on popularity weights
        3. Create directed edges MemoryEntry → Entity
        
        Args:
            G: Graph to add edges to
            entities: List of (entity_id, weight) tuples
        """
        # Extract entity IDs and weights for sampling
        entity_ids = [ent_id for ent_id, _ in entities]
        entity_weights = np.array([weight for _, weight in entities])
        
        # Get all MemoryEntry nodes
        memory_nodes = [
            node for node in G.nodes()
            if G.nodes[node].get('node_type') == 'MemoryEntry'
        ]
        
        # For each MemoryEntry, create MENTIONS edges
        for memory_node in memory_nodes:
            # Sample number of entities to mention
            num_mentions = np.random.randint(
                self.config.mentions_per_event_min,
                self.config.mentions_per_event_max + 1
            )
            
            # Sample entities based on popularity (without replacement)
            # Use multinomial to avoid duplicates
            mentioned_entities = np.random.choice(
                entity_ids,
                size=min(num_mentions, len(entity_ids)),  # Can't mention more than exist
                replace=False,  # No duplicate mentions
                p=entity_weights
            )
            
            # Create edges
            for entity_id in mentioned_entities:
                G.add_edge(
                    memory_node,
                    entity_id,
                    edge_type='MENTIONS',
                    weight=1.0
                )
                
                # Update entity degree counter
                G.nodes[entity_id]['degree'] += 1
    
    def _ensure_connectivity(
        self,
        G: nx.DiGraph,
        entities: List[Tuple[str, float]]
    ):
        """
        Ensure the graph is weakly connected by linking disconnected components.
        
        Strategy: For each disconnected component (session), ensure at least one
        event mentions a "global" entity that appears in multiple components.
        
        Args:
            G: Graph to modify
            entities: List of entities
        """
        # Find weakly connected components
        components = list(nx.weakly_connected_components(G))
        
        if len(components) <= 1:
            return  # Already connected
        
        # Select the most popular entities as "global connectors"
        num_connectors = min(5, len(entities))
        global_entities = [ent_id for ent_id, _ in entities[:num_connectors]]
        
        # For each component, ensure at least one node mentions a global entity
        for component in components:
            # Get MemoryEntry nodes in this component
            memory_nodes = [
                node for node in component
                if G.nodes[node].get('node_type') == 'MemoryEntry'
            ]
            
            if not memory_nodes:
                continue
            
            # Check if component already mentions any global entity
            has_global_connection = False
            for node in memory_nodes:
                neighbors = list(G.successors(node))
                if any(ent in global_entities for ent in neighbors):
                    has_global_connection = True
                    break
            
            # If not, add a connection
            if not has_global_connection:
                # Pick a random memory node in this component
                random_memory = np.random.choice(memory_nodes)
                # Connect to a random global entity
                random_global_entity = np.random.choice(global_entities)
                
                G.add_edge(
                    random_memory,
                    random_global_entity,
                    edge_type='MENTIONS',
                    weight=1.0,
                    connectivity_edge=True  # Mark as synthetic for analysis
                )
                
                G.nodes[random_global_entity]['degree'] += 1
    
    def get_statistics(self, G: nx.DiGraph) -> Dict:
        """
        Compute statistics about entity injection results.
        
        Args:
            G: Graph after entity injection
            
        Returns:
            Dictionary with entity statistics
        """
        # Get entity nodes
        entity_nodes = [
            node for node in G.nodes()
            if G.nodes[node].get('node_type') == 'Entity'
        ]
        
        # Get MENTIONS edges
        mentions_edges = [
            (u, v) for u, v in G.edges()
            if G.edges[u, v].get('edge_type') == 'MENTIONS'
        ]
        
        # Entity degree distribution
        entity_degrees = [G.nodes[ent]['degree'] for ent in entity_nodes]
        
        stats = {
            'num_entities': len(entity_nodes),
            'num_mentions_edges': len(mentions_edges),
            'avg_entity_degree': np.mean(entity_degrees) if entity_degrees else 0,
            'max_entity_degree': max(entity_degrees) if entity_degrees else 0,
            'min_entity_degree': min(entity_degrees) if entity_degrees else 0,
            'std_entity_degree': np.std(entity_degrees) if entity_degrees else 0,
            'is_connected': nx.is_weakly_connected(G),
            'num_components': nx.number_weakly_connected_components(G),
            'graph_density': nx.density(G)
        }
        
        # Degree distribution buckets
        stats['degree_distribution'] = {
            '0': sum(1 for d in entity_degrees if d == 0),
            '1-5': sum(1 for d in entity_degrees if 1 <= d <= 5),
            '6-10': sum(1 for d in entity_degrees if 6 <= d <= 10),
            '11-20': sum(1 for d in entity_degrees if 11 <= d <= 20),
            '21-50': sum(1 for d in entity_degrees if 21 <= d <= 50),
            '51+': sum(1 for d in entity_degrees if d >= 51)
        }
        
        # Top entities (hubs)
        entity_degree_pairs = [(ent, G.nodes[ent]['degree']) for ent in entity_nodes]
        entity_degree_pairs.sort(key=lambda x: x[1], reverse=True)
        stats['top_5_hubs'] = entity_degree_pairs[:5]
        
        return stats
    
    def analyze_heterophily(self, G: nx.DiGraph) -> Dict:
        """
        Analyze heterophily: do similar nodes connect?
        
        Args:
            G: Graph to analyze
            
        Returns:
            Heterophily metrics
        """
        # Count edge types
        edge_type_counts = Counter()
        for u, v in G.edges():
            u_type = G.nodes[u].get('node_type')
            v_type = G.nodes[v].get('node_type')
            edge_type_counts[f"{u_type}→{v_type}"] += 1
        
        total_edges = G.number_of_edges()
        
        metrics = {
            'edge_type_distribution': dict(edge_type_counts),
            'heterophilic_ratio': edge_type_counts.get('MemoryEntry→Entity', 0) / total_edges if total_edges > 0 else 0,
            'temporal_ratio': edge_type_counts.get('MemoryEntry→MemoryEntry', 0) / total_edges if total_edges > 0 else 0
        }
        
        return metrics


def visualize_with_entities(G: nx.DiGraph, max_nodes: int = 100):
    """
    Visualize graph with entities highlighted.
    
    Args:
        G: Graph with entities
        max_nodes: Maximum nodes to display
    """
    import matplotlib.pyplot as plt
    
    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        # Sample evenly from MemoryEntry and Entity nodes
        memory_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'MemoryEntry']
        entity_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity']
        
        sampled_memory = memory_nodes[:int(max_nodes * 0.7)]
        sampled_entity = entity_nodes[:int(max_nodes * 0.3)]
        sampled_nodes = sampled_memory + sampled_entity
        
        G_vis = G.subgraph(sampled_nodes)
    else:
        G_vis = G
    
    # Separate node types for visualization
    memory_nodes = [n for n in G_vis.nodes() if G_vis.nodes[n].get('node_type') == 'MemoryEntry']
    entity_nodes = [n for n in G_vis.nodes() if G_vis.nodes[n].get('node_type') == 'Entity']
    
    # Layout
    pos = nx.spring_layout(G_vis, k=1, iterations=50)
    
    plt.figure(figsize=(14, 10))
    
    # Draw memory nodes (smaller, blue)
    nx.draw_networkx_nodes(
        G_vis, pos,
        nodelist=memory_nodes,
        node_color='skyblue',
        node_size=100,
        label='MemoryEntry',
        alpha=0.7
    )
    
    # Draw entity nodes (larger, red, size by degree)
    entity_sizes = [G_vis.nodes[n]['degree'] * 50 + 100 for n in entity_nodes]
    nx.draw_networkx_nodes(
        G_vis, pos,
        nodelist=entity_nodes,
        node_color='salmon',
        node_size=entity_sizes,
        label='Entity (size=degree)',
        alpha=0.8
    )
    
    # Draw edges with different colors
    next_event_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v].get('edge_type') == 'NEXT_EVENT']
    mentions_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v].get('edge_type') == 'MENTIONS']
    
    nx.draw_networkx_edges(G_vis, pos, edgelist=next_event_edges, edge_color='blue', alpha=0.3, arrows=True, arrowsize=5)
    nx.draw_networkx_edges(G_vis, pos, edgelist=mentions_edges, edge_color='red', alpha=0.2, arrows=True, arrowsize=5)
    
    plt.title(f"Chain-and-Hub Structure (Phase 2)\nBlue: Temporal chains | Red: Entity mentions")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Must import Phase 1 generator
    from backbone_generator import TemporalBackboneGenerator, BackboneConfig
    
    print("=" * 70)
    print("ENTITY INJECTOR - PHASE 2 DEMO")
    print("=" * 70)
    print()
    
    # Step 1: Generate Phase 1 backbone
    print("Step 1: Generating temporal backbone (Phase 1)")
    print("-" * 70)
    backbone_config = BackboneConfig(
        num_sessions=20,
        avg_chain_length=10,
        min_chain_length=5,
        max_chain_length=20
    )
    backbone_gen = TemporalBackboneGenerator(backbone_config)
    G = backbone_gen.generate(seed=42)
    print(f"✓ Created {G.number_of_nodes()} nodes in {G.number_of_edges()} edges")
    print(f"✓ Disconnected components: {nx.number_weakly_connected_components(G)}")
    print()
    
    # Step 2: Inject entities
    print("Step 2: Injecting entities (Phase 2)")
    print("-" * 70)
    entity_config = EntityConfig(
        num_entities=50,
        zipf_alpha=1.5,
        mentions_per_event_min=1,
        mentions_per_event_max=4,
        ensure_connectivity=True
    )
    injector = EntityInjector(entity_config)
    G = injector.inject(G, seed=42)
    print(f"✓ Added {entity_config.num_entities} entities")
    print(f"✓ Total nodes: {G.number_of_nodes()}")
    print(f"✓ Total edges: {G.number_of_edges()}")
    print()
    
    # Step 3: Analyze results
    print("Step 3: Analyzing entity statistics")
    print("-" * 70)
    stats = injector.get_statistics(G)
    print(f"Number of entities: {stats['num_entities']}")
    print(f"MENTIONS edges created: {stats['num_mentions_edges']}")
    print(f"Average entity degree: {stats['avg_entity_degree']:.2f}")
    print(f"Max entity degree (biggest hub): {stats['max_entity_degree']}")
    print(f"Is graph connected: {stats['is_connected']}")
    print(f"Connected components: {stats['num_components']}")
    print()
    
    print("Entity degree distribution:")
    for bucket, count in stats['degree_distribution'].items():
        print(f"  {bucket}: {count} entities")
    print()
    
    print("Top 5 hub entities:")
    for ent_id, degree in stats['top_5_hubs']:
        print(f"  {ent_id}: degree={degree}")
    print()
    
    # Step 4: Analyze heterophily
    print("Step 4: Analyzing heterophily (functional connections)")
    print("-" * 70)
    heterophily = injector.analyze_heterophily(G)
    print("Edge type distribution:")
    for edge_type, count in heterophily['edge_type_distribution'].items():
        print(f"  {edge_type}: {count}")
    print()
    print(f"Heterophilic ratio (MemoryEntry→Entity): {heterophily['heterophilic_ratio']:.2%}")
    print(f"Temporal ratio (MemoryEntry→MemoryEntry): {heterophily['temporal_ratio']:.2%}")
    print()
    
    # Step 5: Sample inspection
    print("Step 5: Sample entity inspection")
    print("-" * 70)
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity']
    for entity in entity_nodes[:3]:
        attrs = G.nodes[entity]
        in_edges = list(G.predecessors(entity))
        print(f"Entity: {entity}")
        print(f"  Popularity: {attrs['popularity']:.4f}")
        print(f"  Degree: {attrs['degree']}")
        print(f"  Rank: {attrs['entity_rank']}")
        print(f"  Connected to {len(in_edges)} memory entries")
        print()
    
    # Step 6: Connectivity analysis
    print("Step 6: Connectivity transformation")
    print("-" * 70)
    print("Before Phase 2 (from backbone):")
    print(f"  Components: {backbone_config.num_sessions} (disconnected sessions)")
    print()
    print("After Phase 2 (with entities):")
    print(f"  Components: {stats['num_components']} (connected via entity hubs!)")
    print()
    
    print("=" * 70)
    print("Phase 2 complete! Next: Phase 3 (Causal Wiring)")
    print("=" * 70)
    
    # Uncomment to visualize
    # visualize_with_entities(G, max_nodes=80)