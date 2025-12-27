"""
This module provides comprehensive validation at research/production standards:
1. Schema Compliance Verification
2. Topological Property Analysis
3. Motif Detection & Counting
4. Statistical Distribution Validation
5. Diversity & Coverage Metrics
6. Comparison with Target Schema
7. Pre-training Readiness Checks

Usage:
    python validate_graphs.py --data_dir data/synthetic --output_dir validation_results
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats
from tqdm import tqdm
import pickle

# Import your generation modules
from src.generation.smsg_pipeline import load_dataset, load_metadata


class GraphValidator:
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to generated dataset
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        
        # Load all splits
        print("Loading dataset...")
        self.train_graphs = load_dataset(data_dir, 'train')
        self.val_graphs = load_dataset(data_dir, 'val')
        self.test_graphs = load_dataset(data_dir, 'test')
        self.all_graphs = self.train_graphs + self.val_graphs + self.test_graphs
        
        self.dataset_meta, self.graphs_meta = load_metadata(data_dir)
        
        print(f"✅ Loaded {len(self.all_graphs)} graphs")
        print(f"   Train: {len(self.train_graphs)}, Val: {len(self.val_graphs)}, Test: {len(self.test_graphs)}")
    
    # ========================================================================
    # VALIDATION 1: SCHEMA COMPLIANCE
    # ========================================================================
    
    def validate_schema_compliance(self) -> Dict:
        """
        Checks:
        - Node types are only 'MemoryEntry' or 'Entity'
        - Edge types are only 'NEXT_EVENT', 'MENTIONS', 'CAUSED_BY'
        - All required attributes present
        - DAG property (no temporal loops)
        - Connectivity (entities link sessions)
        """
        print("\n" + "="*70)
        print("VALIDATION 1: SCHEMA COMPLIANCE")
        print("="*70)
        
        violations = []
        schema_stats = {
            'valid_graphs': 0,
            'invalid_graphs': 0,
            'node_type_violations': [],
            'edge_type_violations': [],
            'missing_attributes': [],
            'dag_violations': [],
            'disconnected_graphs': []
        }
        
        for i, G in enumerate(tqdm(self.all_graphs, desc="Checking schema")):
            graph_valid = True
            
            # Check node types
            for node in G.nodes():
                node_type = G.nodes[node].get('node_type')
                if node_type not in ['MemoryEntry', 'Entity']:
                    violations.append(f"Graph {i}, Node {node}: Invalid type '{node_type}'")
                    schema_stats['node_type_violations'].append(i)
                    graph_valid = False
                    
                # Check required attributes
                if node_type == 'MemoryEntry':
                    required = ['session_id', 'position', 'timestamp']
                    for attr in required:
                        if attr not in G.nodes[node]:
                            violations.append(f"Graph {i}, Node {node}: Missing '{attr}'")
                            schema_stats['missing_attributes'].append((i, node, attr))
                            graph_valid = False
                            
                elif node_type == 'Entity':
                    required = ['popularity', 'degree']
                    for attr in required:
                        if attr not in G.nodes[node]:
                            violations.append(f"Graph {i}, Node {node}: Missing '{attr}'")
                            schema_stats['missing_attributes'].append((i, node, attr))
                            graph_valid = False
            
            # Check edge types
            for u, v in G.edges():
                edge_type = G.edges[u, v].get('edge_type')
                if edge_type not in ['NEXT_EVENT', 'MENTIONS', 'CAUSED_BY']:
                    violations.append(f"Graph {i}, Edge ({u}, {v}): Invalid type '{edge_type}'")
                    schema_stats['edge_type_violations'].append(i)
                    graph_valid = False
            
            # Check DAG property
            if not nx.is_directed_acyclic_graph(G):
                violations.append(f"Graph {i}: Contains cycles (not a DAG)")
                schema_stats['dag_violations'].append(i)
                graph_valid = False
            
            # Check connectivity
            if not nx.is_weakly_connected(G):
                schema_stats['disconnected_graphs'].append(i)
                # Note: This might be OK depending on design, but flag it
            
            if graph_valid:
                schema_stats['valid_graphs'] += 1
            else:
                schema_stats['invalid_graphs'] += 1
        
        # Summary
        print(f"\n✅ Valid graphs: {schema_stats['valid_graphs']}/{len(self.all_graphs)}")
        print(f"❌ Invalid graphs: {schema_stats['invalid_graphs']}/{len(self.all_graphs)}")
        
        if violations:
            print(f"\n⚠️  Found {len(violations)} violations (showing first 10):")
            for v in violations[:10]:
                print(f"   - {v}")
        else:
            print("\n✅ All graphs pass schema compliance!")
        
        # Connectivity warning
        if schema_stats['disconnected_graphs']:
            print(f"\n⚠️  {len(schema_stats['disconnected_graphs'])} graphs are disconnected")
            print("   This may be intentional, but verify entity injection worked correctly.")
        
        self.results['schema_compliance'] = schema_stats
        return schema_stats
    
    # ========================================================================
    # VALIDATION 2: TOPOLOGICAL PROPERTIES
    # ========================================================================
    
    def analyze_topology(self) -> Dict:
        """
        Metrics:
        - Node/edge counts distribution
        - Density distribution
        - Degree distributions (in, out, total)
        - Component analysis
        - Diameter and average path length
        """
        print("\n" + "="*70)
        print("VALIDATION 2: TOPOLOGICAL PROPERTIES")
        print("="*70)
        
        topo_stats = {
            'node_counts': [],
            'edge_counts': [],
            'densities': [],
            'avg_degrees': [],
            'max_degrees': [],
            'num_components': [],
            'diameters': [],
            'avg_path_lengths': []
        }
        
        for G in tqdm(self.all_graphs, desc="Analyzing topology"):
            topo_stats['node_counts'].append(G.number_of_nodes())
            topo_stats['edge_counts'].append(G.number_of_edges())
            topo_stats['densities'].append(nx.density(G))
            
            degrees = [d for n, d in G.degree()]
            topo_stats['avg_degrees'].append(np.mean(degrees))
            topo_stats['max_degrees'].append(max(degrees))
            
            topo_stats['num_components'].append(nx.number_weakly_connected_components(G))
            
            # Diameter (only for connected graphs)
            if nx.is_weakly_connected(G):
                try:
                    diam = nx.diameter(G.to_undirected())
                    topo_stats['diameters'].append(diam)
                except:
                    topo_stats['diameters'].append(None)
                    
                try:
                    avg_path = nx.average_shortest_path_length(G.to_undirected())
                    topo_stats['avg_path_lengths'].append(avg_path)
                except:
                    topo_stats['avg_path_lengths'].append(None)
            else:
                topo_stats['diameters'].append(None)
                topo_stats['avg_path_lengths'].append(None)
        
        # Print summary statistics
        print("\nNode Count Distribution:")
        print(f"   Mean: {np.mean(topo_stats['node_counts']):.1f} ± {np.std(topo_stats['node_counts']):.1f}")
        print(f"   Range: [{min(topo_stats['node_counts'])}, {max(topo_stats['node_counts'])}]")
        
        print("\nEdge Count Distribution:")
        print(f"   Mean: {np.mean(topo_stats['edge_counts']):.1f} ± {np.std(topo_stats['edge_counts']):.1f}")
        print(f"   Range: [{min(topo_stats['edge_counts'])}, {max(topo_stats['edge_counts'])}]")
        
        print("\nDensity Distribution:")
        print(f"   Mean: {np.mean(topo_stats['densities']):.6f} ± {np.std(topo_stats['densities']):.6f}")
        print(f"   Range: [{min(topo_stats['densities']):.6f}, {max(topo_stats['densities']):.6f}]")
        
        print("\nDegree Statistics:")
        print(f"   Avg degree mean: {np.mean(topo_stats['avg_degrees']):.2f}")
        print(f"   Max degree mean: {np.mean(topo_stats['max_degrees']):.1f}")
        print(f"   Max degree range: [{min(topo_stats['max_degrees'])}, {max(topo_stats['max_degrees'])}]")
        
        # Check for diversity
        cv_nodes = np.std(topo_stats['node_counts']) / np.mean(topo_stats['node_counts'])
        cv_edges = np.std(topo_stats['edge_counts']) / np.mean(topo_stats['edge_counts'])
        
        print(f"\nDiversity Check (Coefficient of Variation):")
        print(f"   Nodes: {cv_nodes:.3f} {'✅ Good' if cv_nodes > 0.2 else '⚠️  Low diversity'}")
        print(f"   Edges: {cv_edges:.3f} {'✅ Good' if cv_edges > 0.2 else '⚠️  Low diversity'}")
        
        self.results['topology'] = topo_stats
        return topo_stats
    
    # ========================================================================
    # VALIDATION 3: MOTIF ANALYSIS
    # ========================================================================
    
    def analyze_motifs(self) -> Dict:
        """
        Motifs:
        1. Linear chains (NEXT_EVENT sequences)
        2. Star patterns (Entity hubs)
        3. Triangles (feed-forward loops)
        4. Causal shortcuts (CAUSED_BY bridges)
        """
        print("\n" + "="*70)
        print("VALIDATION 3: MOTIF ANALYSIS")
        print("="*70)
        
        motif_stats = {
            'linear_chains': [],
            'star_patterns': [],
            'triangles': [],
            'causal_shortcuts': [],
            'avg_chain_length': [],
            'max_hub_degree': []
        }
        
        for G in tqdm(self.all_graphs, desc="Counting motifs"):
            # Count linear chains (NEXT_EVENT sequences)
            next_event_edges = [(u, v) for u, v in G.edges() 
                               if G.edges[u, v].get('edge_type') == 'NEXT_EVENT']
            motif_stats['linear_chains'].append(len(next_event_edges))
            
            # Count star patterns (Entity nodes with high degree)
            entity_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity']
            star_count = sum(1 for n in entity_nodes if G.degree(n) >= 5)
            motif_stats['star_patterns'].append(star_count)
            
            if entity_nodes:
                max_hub = max(G.degree(n) for n in entity_nodes)
                motif_stats['max_hub_degree'].append(max_hub)
            else:
                motif_stats['max_hub_degree'].append(0)
            
            # Count triangles
            triangle_count = sum(nx.triangles(G.to_undirected()).values()) // 3
            motif_stats['triangles'].append(triangle_count)
            
            # Count causal shortcuts
            caused_by_edges = [(u, v) for u, v in G.edges() 
                              if G.edges[u, v].get('edge_type') == 'CAUSED_BY']
            motif_stats['causal_shortcuts'].append(len(caused_by_edges))
            
            # Estimate average chain length
            memory_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'MemoryEntry']
            if next_event_edges:
                # Approximate: total memory nodes / number of sessions
                sessions = set(G.nodes[n].get('session_id') for n in memory_nodes)
                avg_chain = len(memory_nodes) / len(sessions) if sessions else 0
                motif_stats['avg_chain_length'].append(avg_chain)
            else:
                motif_stats['avg_chain_length'].append(0)
        
        # Print motif statistics
        print("\nMotif Counts (per graph):")
        print(f"   Linear chains: {np.mean(motif_stats['linear_chains']):.1f} ± {np.std(motif_stats['linear_chains']):.1f}")
        print(f"   Star patterns: {np.mean(motif_stats['star_patterns']):.1f} ± {np.std(motif_stats['star_patterns']):.1f}")
        print(f"   Triangles: {np.mean(motif_stats['triangles']):.1f} ± {np.std(motif_stats['triangles']):.1f}")
        print(f"   Causal shortcuts: {np.mean(motif_stats['causal_shortcuts']):.1f} ± {np.std(motif_stats['causal_shortcuts']):.1f}")
        
        print("\nStructural Metrics:")
        print(f"   Avg chain length: {np.mean(motif_stats['avg_chain_length']):.1f} ± {np.std(motif_stats['avg_chain_length']):.1f}")
        print(f"   Max hub degree: {np.mean(motif_stats['max_hub_degree']):.1f} ± {np.std(motif_stats['max_hub_degree']):.1f}")
        
        # Validate presence of key motifs
        print("\nMotif Presence Check:")
        chains_present = np.mean(motif_stats['linear_chains']) > 0
        stars_present = np.mean(motif_stats['star_patterns']) > 0
        triangles_present = np.mean(motif_stats['triangles']) > 0
        causal_present = np.mean(motif_stats['causal_shortcuts']) > 0
        
        print(f"   ✅ Linear chains: {'Present' if chains_present else '❌ MISSING'}")
        print(f"   ✅ Star patterns: {'Present' if stars_present else '❌ MISSING'}")
        print(f"   ✅ Triangles: {'Present' if triangles_present else '❌ MISSING'}")
        print(f"   ✅ Causal shortcuts: {'Present' if causal_present else '❌ MISSING'}")
        
        self.results['motifs'] = motif_stats
        return motif_stats
    
    # ========================================================================
    # VALIDATION 4: DEGREE DISTRIBUTION
    # ========================================================================
    
    def analyze_degree_distribution(self) -> Dict:
        """
        Expected: Entity nodes follow Zipfian distribution (power-law).
        """
        print("\n" + "="*70)
        print("VALIDATION 4: DEGREE DISTRIBUTION")
        print("="*70)
        
        all_memory_degrees = []
        all_entity_degrees = []
        
        for G in tqdm(self.all_graphs, desc="Collecting degrees"):
            memory_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'MemoryEntry']
            entity_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'Entity']
            
            all_memory_degrees.extend([G.degree(n) for n in memory_nodes])
            all_entity_degrees.extend([G.degree(n) for n in entity_nodes])
        
        print("\nMemory Node Degrees:")
        print(f"   Mean: {np.mean(all_memory_degrees):.2f}")
        print(f"   Median: {np.median(all_memory_degrees):.1f}")
        print(f"   Range: [{min(all_memory_degrees)}, {max(all_memory_degrees)}]")
        
        print("\nEntity Node Degrees:")
        print(f"   Mean: {np.mean(all_entity_degrees):.2f}")
        print(f"   Median: {np.median(all_entity_degrees):.1f}")
        print(f"   Range: [{min(all_entity_degrees)}, {max(all_entity_degrees)}]")
        
        # Test for power-law in entity degrees
        if all_entity_degrees:
            # Fit power law (basic test)
            entity_counts = Counter(all_entity_degrees)
            degrees = sorted(entity_counts.keys())
            degrees = [d for d in degrees if d > 0]
            counts = [entity_counts[d] for d in degrees]
            
            # Log-log regression to check power-law
            log_degrees = np.log(degrees)
            log_counts = np.log(counts)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_counts)
            
            print(f"\nPower-Law Test (Entity Degrees):")
            print(f"   Exponent (slope): {-slope:.3f}")
            print(f"   R² fit: {r_value**2:.3f}")
            print(f"   Expected: α ≈ 1.5 (Zipfian)")
            
            if 1.0 < -slope < 2.5 and r_value**2 > 0.7:
                print("   ✅ Entity degrees follow power-law distribution")
            else:
                print("   ⚠️  Entity degrees may not follow expected power-law")
        
        self.results['degree_distribution'] = {
            'memory_degrees': all_memory_degrees,
            'entity_degrees': all_entity_degrees
        }
        
        return {
            'memory_mean': np.mean(all_memory_degrees),
            'entity_mean': np.mean(all_entity_degrees),
            'power_law_exponent': -slope if all_entity_degrees else None
        }
    
    # ========================================================================
    # VALIDATION 5: HETEROPHILY CHECK
    # ========================================================================
    
    def analyze_heterophily(self) -> Dict:
        """
        Key test: MemoryEntry nodes should NOT preferentially connect to 
        similar MemoryEntry nodes (they connect via temporal/causal logic).
        """
        print("\n" + "="*70)
        print("VALIDATION 5: HETEROPHILY ANALYSIS")
        print("="*70)
        
        edge_type_counts = defaultdict(int)
        
        for G in tqdm(self.all_graphs, desc="Analyzing connections"):
            for u, v in G.edges():
                u_type = G.nodes[u].get('node_type')
                v_type = G.nodes[v].get('node_type')
                edge_type_counts[f"{u_type}→{v_type}"] += 1
        
        total_edges = sum(edge_type_counts.values())
        
        print("\nEdge Type Distribution (across all graphs):")
        for edge_pattern, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
            percentage = count / total_edges * 100
            print(f"   {edge_pattern}: {count:,} ({percentage:.1f}%)")
        
        # Calculate heterophily ratio
        heterophilic_edges = edge_type_counts.get('MemoryEntry→Entity', 0)
        memory_to_memory = edge_type_counts.get('MemoryEntry→MemoryEntry', 0)
        
        heterophily_ratio = heterophilic_edges / total_edges if total_edges > 0 else 0
        temporal_ratio = memory_to_memory / total_edges if total_edges > 0 else 0
        
        print(f"\nHeterophily Metrics:")
        print(f"   Heterophilic edges (MemoryEntry→Entity): {heterophily_ratio:.1%}")
        print(f"   Temporal edges (MemoryEntry→MemoryEntry): {temporal_ratio:.1%}")
        
        if heterophily_ratio > 0.3:
            print("   ✅ Strong heterophilic structure (functional connections)")
        else:
            print("   ⚠️  Low heterophily - may be too homophilic")
        
        self.results['heterophily'] = {
            'edge_type_counts': dict(edge_type_counts),
            'heterophily_ratio': heterophily_ratio,
            'temporal_ratio': temporal_ratio
        }
        
        return self.results['heterophily']
    
    # ========================================================================
    # VALIDATION 6: PARAMETER DIVERSITY
    # ========================================================================
    
    def analyze_parameter_diversity(self) -> Dict:
        """
        Critical for generalization: Model must see diverse topologies.
        """
        print("\n" + "="*70)
        print("VALIDATION 6: PARAMETER DIVERSITY")
        print("="*70)
        
        params_collected = {
            'num_sessions': [],
            'num_entities': [],
            'avg_chain_length': [],
            'causal_prob_base': []
        }
        
        for meta in self.graphs_meta:
            params = meta['params']
            params_collected['num_sessions'].append(params['num_sessions'])
            params_collected['num_entities'].append(params['num_entities'])
            params_collected['avg_chain_length'].append(params['avg_chain_length'])
            params_collected['causal_prob_base'].append(params['causal_prob_base'])
        
        print("\nParameter Ranges:")
        for param_name, values in params_collected.items():
            print(f"\n   {param_name}:")
            print(f"      Range: [{min(values)}, {max(values)}]")
            print(f"      Mean: {np.mean(values):.2f} ± {np.std(values):.2f}")
            print(f"      CV: {np.std(values)/np.mean(values):.3f}")
            
            # Check if truly varied
            unique_vals = len(set(values))
            possible_values = max(values) - min(values) + 1
            coverage = unique_vals / possible_values
            if coverage < 0.8:
                print(f"      ⚠️  Only {coverage:.2f} of possible values - low diversity!")
            else:
                print(f"      ✅ {coverage:.2f} of possible values - good diversity")
        
        self.results['parameter_diversity'] = params_collected
        return params_collected
    
    # ========================================================================
    # VALIDATION 7: PRE-TRAINING READINESS
    # ========================================================================
    
    def check_pretraining_readiness(self) -> Dict:
        """
        Requirements:
        1. All graphs are valid
        2. Sufficient diversity
        3. Clear structural patterns
        4. No degenerate cases
        """
        print("\n" + "="*70)
        print("VALIDATION 7: PRE-TRAINING READINESS")
        print("="*70)
        
        readiness = {
            'schema_valid': True,
            'diversity_ok': True,
            'motifs_present': True,
            'no_degenerates': True,
            'ready': False
        }
        
        # Check 1: Schema compliance
        if self.results.get('schema_compliance'):
            invalid_count = self.results['schema_compliance']['invalid_graphs']
            if invalid_count > 0:
                print(f"   ❌ {invalid_count} graphs failed schema validation")
                readiness['schema_valid'] = False
            else:
                print("   ✅ All graphs pass schema validation")
        
        # Check 2: Diversity
        if self.results.get('topology'):
            cv_nodes = np.std(self.results['topology']['node_counts']) / np.mean(self.results['topology']['node_counts'])
            if cv_nodes < 0.15:
                print(f"   ⚠️  Low topology diversity (CV={cv_nodes:.3f})")
                readiness['diversity_ok'] = False
            else:
                print(f"   ✅ Good topology diversity (CV={cv_nodes:.3f})")
        
        # Check 3: Motifs present
        if self.results.get('motifs'):
            chains = np.mean(self.results['motifs']['linear_chains'])
            stars = np.mean(self.results['motifs']['star_patterns'])
            triangles = np.mean(self.results['motifs']['triangles'])
            
            if chains == 0 or stars == 0:
                print("   ❌ Missing essential motifs (chains or hubs)")
                readiness['motifs_present'] = False
            else:
                print("   ✅ Essential motifs present")
                print(f"      Chains: {chains:.1f}, Stars: {stars:.1f}, Triangles: {triangles:.1f}")
        
        # Check 4: No degenerate graphs
        if self.results.get('topology'):
            min_nodes = min(self.results['topology']['node_counts'])
            min_edges = min(self.results['topology']['edge_counts'])
            
            if min_nodes < 10 or min_edges < 5:
                print(f"   ⚠️  Some graphs are very small (min_nodes={min_nodes}, min_edges={min_edges})")
                readiness['no_degenerates'] = False
            else:
                print("   ✅ No degenerate graphs detected")
        
        # Overall readiness
        readiness['ready'] = all([
            readiness['schema_valid'],
            readiness['diversity_ok'],
            readiness['motifs_present'],
            readiness['no_degenerates']
        ])
        
        print("\n" + "="*70)
        if readiness['ready']:
            print("✅ DATASET IS READY FOR GRAPHSAGE PRE-TRAINING")
        else:
            print("❌ DATASET NOT READY - ADDRESS ISSUES ABOVE")
        print("="*70)
        
        self.results['readiness'] = readiness
        return readiness
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def create_visualizations(self, output_dir: Path):
        """
        Create comprehensive visualization report.
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Figure 1: Topology distributions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Graph Topology Distributions', fontsize=16, fontweight='bold')
        
        # Node count distribution
        axes[0, 0].hist(self.results['topology']['node_counts'], bins=30, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Node Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Node Count Distribution')
        axes[0, 0].axvline(np.mean(self.results['topology']['node_counts']), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Edge count distribution
        axes[0, 1].hist(self.results['topology']['edge_counts'], bins=30, 
                       color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Edge Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Edge Count Distribution')
        axes[0, 1].axvline(np.mean(self.results['topology']['edge_counts']), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Density distribution
        axes[1, 0].hist(self.results['topology']['densities'], bins=30, 
                       color='mediumseagreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Graph Density')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Density Distribution')
        axes[1, 0].axvline(np.mean(self.results['topology']['densities']), 
                          color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # Max degree distribution
        axes[1, 1].hist(self.results['topology']['max_degrees'], bins=30, 
                       color='mediumpurple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Max Degree')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Max Degree Distribution')
        axes[1, 1].axvline(np.mean(self.results['topology']['max_degrees']), 
                          color='red', linestyle='--', label='Mean')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'topology_distributions.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: topology_distributions.png")
        plt.close()
        
        # Figure 2: Degree distributions (power-law check)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Degree Distributions', fontsize=16, fontweight='bold')
        
        # Memory node degrees
        axes[0].hist(self.results['degree_distribution']['memory_degrees'], 
                    bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Degree')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('MemoryEntry Node Degrees')
        axes[0].set_yscale('log')
        
        # Entity node degrees (log-log for power-law)
        axes[1].hist(self.results['degree_distribution']['entity_degrees'], 
                    bins=50, color='salmon', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Degree')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Entity Node Degrees (Should be Power-Law)')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'degree_distributions.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: degree_distributions.png")
        plt.close()
        
        # Figure 3: Motif counts
        fig, ax = plt.subplots(figsize=(10, 6))
        
        motif_names = ['Linear Chains', 'Star Patterns', 'Triangles', 'Causal Shortcuts']
        motif_means = [
            np.mean(self.results['motifs']['linear_chains']),
            np.mean(self.results['motifs']['star_patterns']),
            np.mean(self.results['motifs']['triangles']),
            np.mean(self.results['motifs']['causal_shortcuts'])
        ]
        motif_stds = [
            np.std(self.results['motifs']['linear_chains']),
            np.std(self.results['motifs']['star_patterns']),
            np.std(self.results['motifs']['triangles']),
            np.std(self.results['motifs']['causal_shortcuts'])
        ]
        
        colors = ['blue', 'red', 'green', 'purple']
        x_pos = np.arange(len(motif_names))
        
        ax.bar(x_pos, motif_means, yerr=motif_stds, alpha=0.7, 
              color=colors, edgecolor='black', capsize=5)
        ax.set_xlabel('Motif Type', fontsize=12)
        ax.set_ylabel('Average Count per Graph', fontsize=12)
        ax.set_title('Memory-R1 Motif Presence', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(motif_names)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'motif_counts.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: motif_counts.png")
        plt.close()
        
        print(f"\n✅ All visualizations saved to {output_dir}")
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    
    def generate_report(self, output_dir: Path):
        """
        Generate comprehensive validation report in JSON and TXT.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / 'validation_report.json'
        with open(json_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else 
                           int(v) if isinstance(v, np.integer) else
                           float(v) if isinstance(v, np.floating) else v)
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Saved: {json_path}")
        
        # Save human-readable TXT
        txt_path = output_dir / 'validation_report.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MEMORY-R1 SYNTHETIC GRAPH VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {self.data_dir}\n")
            f.write(f"Total graphs: {len(self.all_graphs)}\n")
            f.write(f"Train/Val/Test: {len(self.train_graphs)}/{len(self.val_graphs)}/{len(self.test_graphs)}\n\n")
            
            # Write all results
            for section, data in self.results.items():
                f.write("-"*70 + "\n")
                f.write(f"{section.upper().replace('_', ' ')}\n")
                f.write("-"*70 + "\n")
                f.write(str(data) + "\n\n")
        
        print(f"✅ Saved: {txt_path}")
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_all_validations(self, output_dir: str = 'validation_results'):
        """
        Run complete validation suite.
        """
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE VALIDATION")
        print("="*70)
        
        # Run all validations
        self.validate_schema_compliance()
        self.analyze_topology()
        self.analyze_motifs()
        self.analyze_degree_distribution()
        self.analyze_heterophily()
        self.analyze_parameter_diversity()
        self.check_pretraining_readiness()
        
        # Generate outputs
        output_dir = Path(output_dir)
        self.create_visualizations(output_dir)
        self.generate_report(output_dir)
        
        print("\n" + "="*70)
        print("✅ VALIDATION COMPLETE")
        print(f"📊 Results saved to: {output_dir}")
        print("="*70)
        
        return self.results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Memory-R1 synthetic graphs')
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                       help='Directory containing generated graphs')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='Directory to save validation results')
    
    args = parser.parse_args()
    
    # Run validation
    validator = GraphValidator(args.data_dir)
    results = validator.run_all_validations(args.output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if results.get('readiness', {}).get('ready', False):
        print("\n🎉 SUCCESS! Your dataset is ready for GraphSAGE pre-training.")
        print("\nNext steps:")
        print("  1. Review visualizations in validation_results/")
        print("  2. Implement feature extraction (degree, LDP, RWPE)")
        print("  3. Build GraphSAGE model")
        print("  4. Start pre-training with link prediction")
    else:
        print("\n⚠️  Dataset needs improvements before pre-training.")
        print("\nReview the validation report for specific issues.")
        print("Common fixes:")
        print("  - Adjust generation parameters for more diversity")
        print("  - Ensure entity injection connects all sessions")
        print("  - Tune causal edge probability")