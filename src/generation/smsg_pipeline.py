"""
Schema-Mimetic Synthetic Generation (SMSG) Pipeline

This module orchestrates all three phases of synthetic graph generation for Memory-R1:
Phase 1: Temporal Backbone (event chains)
Phase 2: Entity Injection (hub structures)
Phase 3: Causal Wiring (triangular motifs)

The pipeline generates multiple graphs with parameter variation to ensure GraphSAGE
learns robust structural patterns across diverse topologies.
"""

import networkx as nx
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse
from datetime import datetime

# Import all phase generators
from .backbone_generator import TemporalBackboneGenerator, BackboneConfig
from .entity_injection import EntityInjector, EntityConfig
from .causal_wiring import CausalWiring, CausalConfig


@dataclass
class SMSGConfig:
    """Complete configuration for SMSG pipeline.
    
    This combines all three phase configurations plus dataset-level settings.
    """
    # Dataset settings
    num_graphs: int = 1000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    
    # Phase 1: Backbone
    num_sessions_min: int = 10
    num_sessions_max: int = 100
    avg_chain_length_min: float = 5.0
    avg_chain_length_max: float = 20.0
    
    # Phase 2: Entities
    num_entities_min: int = 20
    num_entities_max: int = 200
    zipf_alpha_min: float = 1.0
    zipf_alpha_max: float = 2.0
    mentions_per_event_min: int = 2
    mentions_per_event_max: int = 6
    
    # Phase 3: Causality
    causal_prob_base_min: float = 0.01
    causal_prob_base_max: float = 0.1
    causal_boost_shared_entity: float = 3.0
    max_causal_distance: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert self.num_graphs > 0, "Must generate at least 1 graph"


class SMSGPipeline:
    """  
    This class handles:
    1. Parameter sampling (creates diverse graphs)
    2. Three-phase generation (backbone → entities → causality)
    3. Dataset splitting (train/val/test)
    4. Serialization and metadata tracking
    5. Statistics collection
    
    The pipeline ensures diversity by sampling parameters from ranges rather
    than using fixed values, forcing GraphSAGE to generalize across topologies.
    """
    
    def __init__(self, config: SMSGConfig):
        """ 
        Args:
            config: Complete configuration for generation
        """
        self.config = config
        self.graphs = []
        self.metadata = []
        
        # Set random seed for reproducibility
        np.random.seed(config.seed)
        
    def generate_dataset(self) -> Tuple[List[nx.DiGraph], Dict]:
        """
        Returns:
            Tuple of (list of graphs, dataset metadata)
        """
        print(f"Generating {self.config.num_graphs} synthetic Memory-R1 graphs...")
        print("=" * 70)
        
        for i in tqdm(range(self.config.num_graphs), desc="Generating graphs"):
            # Sample random parameters for this graph
            params = self._sample_parameters(i)
            
            # Generate graph through all 3 phases
            G = self._generate_single_graph(params, graph_id=i)
            
            # Store graph and metadata
            self.graphs.append(G)
            self.metadata.append({
                'graph_id': i,
                'params': params,
                'stats': self._collect_graph_stats(G)
            })
        
        # Create dataset metadata
        dataset_metadata = self._create_dataset_metadata()
        
        print("Generation complete!")
        return self.graphs, dataset_metadata
    
    def _sample_parameters(self, graph_id: int) -> Dict:
        """
        Sample random parameters for a single graph.

        Args:
            graph_id: Unique identifier for this graph
            
        Returns:
            Dictionary of sampled parameters
        """
        return {
            # Phase 1: Backbone
            'num_sessions': np.random.randint(
                self.config.num_sessions_min,
                self.config.num_sessions_max + 1
            ),
            'avg_chain_length': np.random.uniform(
                self.config.avg_chain_length_min,
                self.config.avg_chain_length_max
            ),
            'chain_length_std': np.random.uniform(2.0, 5.0),
            
            # Phase 2: Entities
            'num_entities': np.random.randint(
                self.config.num_entities_min,
                self.config.num_entities_max + 1
            ),
            'zipf_alpha': np.random.uniform(
                self.config.zipf_alpha_min,
                self.config.zipf_alpha_max
            ),
            'mentions_per_event_min': self.config.mentions_per_event_min,
            'mentions_per_event_max': self.config.mentions_per_event_max,
            
            # Phase 3: Causality
            'causal_prob_base': np.random.uniform(
                self.config.causal_prob_base_min,
                self.config.causal_prob_base_max
            ),
            'causal_boost_shared_entity': self.config.causal_boost_shared_entity,
            'max_causal_distance': self.config.max_causal_distance,
            
            # Seed for this graph (derived from global seed)
            'seed': self.config.seed + graph_id
        }
    
    def _generate_single_graph(self, params: Dict, graph_id: int) -> nx.DiGraph:
        """
  
        Args:
            params: Sampled parameters for this graph
            graph_id: Unique identifier
            
        Returns:
            Complete synthetic graph
        """
        # Phase 1: Temporal Backbone
        backbone_config = BackboneConfig(
            num_sessions=params['num_sessions'],
            avg_chain_length=params['avg_chain_length'],
            chain_length_std=params['chain_length_std']
        )
        backbone_gen = TemporalBackboneGenerator(backbone_config)
        G = backbone_gen.generate(seed=params['seed'])
        
        # Phase 2: Entity Injection
        entity_config = EntityConfig(
            num_entities=params['num_entities'],
            zipf_alpha=params['zipf_alpha'],
            mentions_per_event_min=params['mentions_per_event_min'],
            mentions_per_event_max=params['mentions_per_event_max'],
            ensure_connectivity=True
        )
        injector = EntityInjector(entity_config)
        G = injector.inject(G, seed=params['seed'])
        
        # Phase 3: Causal Wiring
        causal_config = CausalConfig(
            causal_prob_base=params['causal_prob_base'],
            causal_boost_shared_entity=params['causal_boost_shared_entity'],
            max_causal_distance=params['max_causal_distance'],
            prefer_triangles=True,
            triangle_boost=2.0
        )
        wiring = CausalWiring(causal_config)
        G = wiring.wire(G, seed=params['seed'])
        
        # Add graph-level metadata
        G.graph['graph_id'] = graph_id
        G.graph['generation_timestamp'] = datetime.now().isoformat()
        G.graph['generation_params'] = params
        
        return G
    
    def _collect_graph_stats(self, G: nx.DiGraph) -> Dict:
        """  
        Args:
            G: Generated graph
            
        Returns:
            Dictionary of statistics
        """
        # Basic stats
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'is_dag': nx.is_directed_acyclic_graph(G)
        }
        
        # Node type counts
        node_types = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats['node_types'] = node_types
        
        # Edge type counts
        edge_types = {}
        for u, v in G.edges():
            edge_type = G.edges[u, v].get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats['edge_types'] = edge_types
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        stats['degree_stats'] = {
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees))
        }
        
        return stats
    
    def _create_dataset_metadata(self) -> Dict:
        """        
        Returns:
            Dataset-level metadata
        """
        # Aggregate statistics
        all_stats = [m['stats'] for m in self.metadata]
        
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'num_graphs': len(self.graphs),
            'splits': {
                'train': int(self.config.num_graphs * self.config.train_ratio),
                'val': int(self.config.num_graphs * self.config.val_ratio),
                'test': int(self.config.num_graphs * self.config.test_ratio)
            },
            'aggregate_stats': {
                'total_nodes': sum(s['num_nodes'] for s in all_stats),
                'total_edges': sum(s['num_edges'] for s in all_stats),
                'avg_nodes_per_graph': float(np.mean([s['num_nodes'] for s in all_stats])),
                'avg_edges_per_graph': float(np.mean([s['num_edges'] for s in all_stats])),
                'avg_density': float(np.mean([s['density'] for s in all_stats])),
                'all_connected': all(s['is_connected'] for s in all_stats),
                'all_dags': all(s['is_dag'] for s in all_stats)
            }
        }
        
        return metadata
    
    def split_dataset(
        self
    ) -> Tuple[List[nx.DiGraph], List[nx.DiGraph], List[nx.DiGraph]]:
        """  
        Returns:
            Tuple of (train_graphs, val_graphs, test_graphs)
        """
        n = len(self.graphs)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        
        train = self.graphs[:n_train]
        val = self.graphs[n_train:n_train + n_val]
        test = self.graphs[n_train + n_val:]
        
        return train, val, test
    
    def save_dataset(self, output_dir: Path):
        """
        Saves:
        - Individual graphs as pickled NetworkX objects
        - Dataset metadata as JSON
        - Per-graph metadata as JSON
        
        Args:
            output_dir: Directory to save dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        test_dir = output_dir / 'test'
        
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(exist_ok=True)
        
        print(f"\nSaving dataset to {output_dir}...")
        
        # Split dataset
        train, val, test = self.split_dataset()
        
        # Save each split
        self._save_split(train, train_dir, 'train')
        self._save_split(val, val_dir, 'val')
        self._save_split(test, test_dir, 'test')
        
        # Save dataset metadata
        dataset_metadata = self._create_dataset_metadata()
        with open(output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Save per-graph metadata
        with open(output_dir / 'graphs_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"   Dataset saved successfully!")
        print(f"   Train: {len(train)} graphs")
        print(f"   Val: {len(val)} graphs")
        print(f"   Test: {len(test)} graphs")
    
    def _save_split(self, graphs: List[nx.DiGraph], output_dir: Path, split_name: str):
        """
        Save a single split to disk.
        
        Args:
            graphs: List of graphs to save
            output_dir: Directory for this split
            split_name: Name of split ('train', 'val', 'test')
        """
        for i, G in enumerate(tqdm(graphs, desc=f"Saving {split_name}")):
            # Save as pickle (preserves all NetworkX metadata)
            graph_path = output_dir / f'graph_{i:04d}.gpickle'
            with open(graph_path, 'wb') as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def print_summary(self):
        """Print summary statistics about the generated dataset."""
        metadata = self._create_dataset_metadata()
        
        print("\n" + "=" * 70)
        print("DATASET GENERATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal graphs generated: {metadata['num_graphs']}")
        print(f"  Train: {metadata['splits']['train']}")
        print(f"  Val: {metadata['splits']['val']}")
        print(f"  Test: {metadata['splits']['test']}")
        
        print(f"\nAggregate Statistics:")
        print(f"  Total nodes: {metadata['aggregate_stats']['total_nodes']:,}")
        print(f"  Total edges: {metadata['aggregate_stats']['total_edges']:,}")
        print(f"  Avg nodes per graph: {metadata['aggregate_stats']['avg_nodes_per_graph']:.1f}")
        print(f"  Avg edges per graph: {metadata['aggregate_stats']['avg_edges_per_graph']:.1f}")
        print(f"  Avg density: {metadata['aggregate_stats']['avg_density']:.6f}")
        
        print(f"\nTopological Properties:")
        print(f"  All graphs connected: {metadata['aggregate_stats']['all_connected']}")
        print(f"  All graphs are DAGs: {metadata['aggregate_stats']['all_dags']}")
        
        # Parameter ranges
        print(f"\nParameter Ranges:")
        print(f"  Sessions: [{self.config.num_sessions_min}, {self.config.num_sessions_max}]")
        print(f"  Chain length: [{self.config.avg_chain_length_min:.1f}, {self.config.avg_chain_length_max:.1f}]")
        print(f"  Entities: [{self.config.num_entities_min}, {self.config.num_entities_max}]")
        print(f"  Causal prob: [{self.config.causal_prob_base_min:.3f}, {self.config.causal_prob_base_max:.3f}]")
        
        print("\n" + "=" * 70)


def load_dataset(data_dir: Path, split: str = 'train') -> List[nx.DiGraph]:
    """
    Load a saved dataset split.
    
    Args:
        data_dir: Root directory of saved dataset
        split: Which split to load ('train', 'val', 'test')
        
    Returns:
        List of loaded graphs
    """
    split_dir = Path(data_dir) / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    graph_files = sorted(split_dir.glob('graph_*.gpickle'))
    
    graphs = []
    for graph_path in tqdm(graph_files, desc=f"Loading {split}"):
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            graphs.append(G)
    
    return graphs


def load_metadata(data_dir: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load dataset metadata.
    
    Args:
        data_dir: Root directory of saved dataset
        
    Returns:
        Tuple of (dataset_metadata, graphs_metadata)
    """
    data_dir = Path(data_dir)
    
    with open(data_dir / 'dataset_metadata.json', 'r') as f:
        dataset_metadata = json.load(f)
    
    with open(data_dir / 'graphs_metadata.json', 'r') as f:
        graphs_metadata = json.load(f)
    
    return dataset_metadata, graphs_metadata



def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic Memory-R1 graphs using SMSG pipeline'
    )
    
    # Dataset settings
    parser.add_argument('--num_graphs', type=int, default=1000,
                       help='Total number of graphs to generate')
    parser.add_argument('--output', type=str, default='data/synthetic',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Proportion of data for testing')
    
    # Parameter ranges
    parser.add_argument('--sessions_min', type=int, default=10,
                       help='Minimum number of sessions per graph')
    parser.add_argument('--sessions_max', type=int, default=100,
                       help='Maximum number of sessions per graph')
    parser.add_argument('--entities_min', type=int, default=20,
                       help='Minimum number of entities per graph')
    parser.add_argument('--entities_max', type=int, default=200,
                       help='Maximum number of entities per graph')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SMSGConfig(
        num_graphs=args.num_graphs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_sessions_min=args.sessions_min,
        num_sessions_max=args.sessions_max,
        num_entities_min=args.entities_min,
        num_entities_max=args.entities_max
    )
    
    # Run pipeline
    pipeline = SMSGPipeline(config)
    graphs, metadata = pipeline.generate_dataset()
    
    # Print summary
    pipeline.print_summary()
    
    # Save dataset
    pipeline.save_dataset(Path(args.output))
    
    print("\n🎉 Complete! Dataset ready for feature extraction and training.")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode: generate small dataset
        print("=" * 70)
        print("SMSG PIPELINE DEMO (generating 10 graphs)")
        print("=" * 70)
        print()
        
        # Create small config for demo
        config = SMSGConfig(
            num_graphs=10,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            num_sessions_min=10,
            num_sessions_max=30,
            num_entities_min=20,
            num_entities_max=60
        )
        
        # Generate dataset
        pipeline = SMSGPipeline(config)
        graphs, metadata = pipeline.generate_dataset()
        
        # Print summary
        pipeline.print_summary()
        
        # Demonstrate saving
        output_dir = Path('data/synthetic_demo')
        pipeline.save_dataset(output_dir)
        
        print("\nDemonstrating dataset loading...")
        train_graphs = load_dataset(output_dir, 'train')
        print(f"✅ Loaded {len(train_graphs)} training graphs")
        
        dataset_meta, graphs_meta = load_metadata(output_dir)
        print(f"✅ Loaded metadata for {len(graphs_meta)} graphs")
        
        print("\n" + "=" * 70)
        print("Demo complete! For production, run:")
        print("  python -m src.generation.smsg_pipeline --num_graphs 1000 --output data/synthetic")
        print("=" * 70)