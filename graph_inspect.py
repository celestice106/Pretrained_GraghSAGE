"""
Graph Inspection and Visualization Tools

Use this script to inspect and visualize your generated synthetic graphs.

Usage:
    python inspect_graphs.py
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from collections import Counter
from src.generation.smsg_pipeline import load_dataset, load_metadata


def analyze_output_summary(data_dir='data/synthetic_demo'):
    """
    Analyze the generation summary and explain what each metric means.
    """
    print("=" * 70)
    print("UNDERSTANDING YOUR GENERATION OUTPUT")
    print("=" * 70)
    print()
    
    # Load metadata
    dataset_meta, graphs_meta = load_metadata(data_dir)
    
    print("📊 DATASET STATISTICS")
    print("-" * 70)
    print(f"Total graphs: {dataset_meta['num_graphs']}")
    print(f"  ✓ Train: {dataset_meta['splits']['train']}")
    print(f"  ✓ Val: {dataset_meta['splits']['val']}")
    print(f"  ✓ Test: {dataset_meta['splits']['test']}")
    print()
    
    print("📈 AGGREGATE STATISTICS")
    print("-" * 70)
    agg = dataset_meta['aggregate_stats']
    print(f"Total nodes: {agg['total_nodes']:,}")
    print(f"Total edges: {agg['total_edges']:,}")
    print(f"Avg nodes per graph: {agg['avg_nodes_per_graph']:.1f}")
    print(f"  → Interpretation: Each graph has ~{agg['avg_nodes_per_graph']:.0f} nodes")
    print(f"    (mix of MemoryEntry and Entity nodes)")
    print()
    
    print(f"Avg edges per graph: {agg['avg_edges_per_graph']:.1f}")
    print(f"  → Interpretation: Each graph has ~{agg['avg_edges_per_graph']:.0f} edges")
    print(f"    (NEXT_EVENT + MENTIONS + CAUSED_BY)")
    print()
    
    print(f"Avg density: {agg['avg_density']:.6f}")
    print(f"  → Interpretation: Graphs are sparse (good for Memory-R1!)")
    print(f"    Density = {agg['avg_density']:.6f} means ~{agg['avg_density']*100:.4f}% of possible edges exist")
    print()
    
    print("🔍 TOPOLOGICAL PROPERTIES")
    print("-" * 70)
    print(f"All graphs connected: {agg['all_connected']}")
    if not agg['all_connected']:
        print("  ⚠️  WARNING: Some graphs are NOT connected!")
        print("     This might indicate an issue with entity injection.")
        print("     Let's investigate which graphs are disconnected...")
        
        # Find disconnected graphs
        disconnected = []
        for meta in graphs_meta:
            if not meta['stats']['is_connected']:
                disconnected.append(meta['graph_id'])
        
        print(f"     Disconnected graphs: {disconnected}")
        print(f"     Count: {len(disconnected)} out of {len(graphs_meta)}")
    else:
        print("  ✓ Good! All graphs are weakly connected.")
    print()
    
    print(f"All graphs are DAGs: {agg['all_dags']}")
    if agg['all_dags']:
        print("  ✓ Good! All graphs are Directed Acyclic Graphs (no time loops)")
    else:
        print("  ⚠️  WARNING: Some graphs have cycles!")
    print()
    
    return dataset_meta, graphs_meta


def inspect_single_graph(G, graph_id=0):
    """
    Deep dive into a single graph's structure.
    """
    print("=" * 70)
    print(f"INSPECTING GRAPH {graph_id}")
    print("=" * 70)
    print()
    
    # Basic info
    print("📋 BASIC INFO")
    print("-" * 70)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    print(f"Is connected: {nx.is_weakly_connected(G)}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print()
    
    # Node type breakdown
    print("🔢 NODE TYPES")
    print("-" * 70)
    node_types = Counter(G.nodes[n]['node_type'] for n in G.nodes())
    for node_type, count in node_types.items():
        percentage = count / G.number_of_nodes() * 100
        print(f"{node_type}: {count} ({percentage:.1f}%)")
    print()
    
    # Edge type breakdown
    print("🔗 EDGE TYPES")
    print("-" * 70)
    edge_types = Counter(G.edges[u, v]['edge_type'] for u, v in G.edges())
    for edge_type, count in edge_types.items():
        percentage = count / G.number_of_edges() * 100
        print(f"{edge_type}: {count} ({percentage:.1f}%)")
    print()
    
    # Degree distribution
    print("📊 DEGREE DISTRIBUTION")
    print("-" * 70)
    degrees = [d for n, d in G.degree()]
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Mean degree: {np.mean(degrees):.2f}")
    print(f"Std degree: {np.std(degrees):.2f}")
    print()
    
    # Degree histogram
    print("Degree histogram:")
    degree_bins = [0, 2, 5, 10, 20, 50, 100, float('inf')]
    degree_labels = ['0-1', '2-4', '5-9', '10-19', '20-49', '50-99', '100+']
    degree_counts = np.histogram(degrees, bins=degree_bins)[0]
    for label, count in zip(degree_labels, degree_counts):
        if count > 0:
            print(f"  {label}: {count} nodes")
    print()
    
    # Hub analysis
    print("🎯 HUB ANALYSIS (Top 5 highest degree nodes)")
    print("-" * 70)
    degree_list = [(n, d) for n, d in G.degree()]
    degree_list.sort(key=lambda x: x[1], reverse=True)
    
    for i, (node, degree) in enumerate(degree_list[:5]):
        node_type = G.nodes[node]['node_type']
        print(f"{i+1}. {node} (type: {node_type}, degree: {degree})")
    print()
    
    # Connectivity analysis
    if not nx.is_weakly_connected(G):
        print("⚠️  CONNECTIVITY ISSUE")
        print("-" * 70)
        components = list(nx.weakly_connected_components(G))
        print(f"Number of disconnected components: {len(components)}")
        print(f"Component sizes: {[len(c) for c in components]}")
        print()
        print("This suggests entity injection didn't fully connect the graph.")
        print("Possible causes:")
        print("  - Too few entities")
        print("  - Entities not distributed across sessions")
        print("  - ensure_connectivity flag not working")
        print()


def visualize_graph_structure(G, graph_id=0, save_path=None):
    """
    Create comprehensive visualization of graph structure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Graph {graph_id} Structure Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data
    memory_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'MemoryEntry']
    entity_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Entity']
    
    # Sample for visualization (too many nodes is cluttered)
    max_display = 80
    if len(memory_nodes) > max_display * 0.7:
        memory_sample = np.random.choice(memory_nodes, int(max_display * 0.7), replace=False).tolist()
    else:
        memory_sample = memory_nodes
    
    if len(entity_nodes) > max_display * 0.3:
        entity_sample = np.random.choice(entity_nodes, int(max_display * 0.3), replace=False).tolist()
    else:
        entity_sample = entity_nodes
    
    G_vis = G.subgraph(memory_sample + entity_sample)
    
    # Layout
    pos = nx.spring_layout(G_vis, k=1.5, iterations=50, seed=42)
    
    # 1. Full graph structure
    ax1 = axes[0, 0]
    ax1.set_title('Graph Structure (blue=MemoryEntry, red=Entity)', fontsize=12, fontweight='bold')
    
    # Draw nodes
    nx.draw_networkx_nodes(G_vis, pos, nodelist=[n for n in G_vis.nodes() if n in memory_sample],
                          node_color='skyblue', node_size=100, alpha=0.7, ax=ax1)
    
    entity_sizes = [G_vis.nodes[n]['degree'] * 30 + 100 for n in G_vis.nodes() if n in entity_sample]
    nx.draw_networkx_nodes(G_vis, pos, nodelist=[n for n in G_vis.nodes() if n in entity_sample],
                          node_color='salmon', node_size=entity_sizes, alpha=0.8, ax=ax1)
    
    # Draw edges by type
    next_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v]['edge_type'] == 'NEXT_EVENT']
    mentions_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v]['edge_type'] == 'MENTIONS']
    causal_edges = [(u, v) for u, v in G_vis.edges() if G_vis.edges[u, v]['edge_type'] == 'CAUSED_BY']
    
    nx.draw_networkx_edges(G_vis, pos, edgelist=next_edges, edge_color='blue',
                          alpha=0.3, width=1.5, arrows=True, arrowsize=8, ax=ax1)
    nx.draw_networkx_edges(G_vis, pos, edgelist=mentions_edges, edge_color='red',
                          alpha=0.2, width=1.0, arrows=True, arrowsize=6, ax=ax1)
    nx.draw_networkx_edges(G_vis, pos, edgelist=causal_edges, edge_color='green',
                          alpha=0.5, width=2.0, style='dashed', arrows=True, arrowsize=10, ax=ax1)
    
    ax1.axis('off')
    
    # 2. Degree distribution
    ax2 = axes[0, 1]
    ax2.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    
    degrees = [d for n, d in G.degree()]
    ax2.hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(degrees):.1f}')
    ax2.axvline(np.median(degrees), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(degrees):.1f}')
    ax2.set_xlabel('Degree', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Edge type distribution
    ax3 = axes[1, 0]
    ax3.set_title('Edge Type Distribution', fontsize=12, fontweight='bold')
    
    edge_types = Counter(G.edges[u, v]['edge_type'] for u, v in G.edges())
    colors = {'NEXT_EVENT': 'blue', 'MENTIONS': 'red', 'CAUSED_BY': 'green'}
    
    labels = list(edge_types.keys())
    values = list(edge_types.values())
    bar_colors = [colors.get(label, 'gray') for label in labels]
    
    bars = ax3.bar(labels, values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    total = sum(values)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}\n({value/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Node type distribution
    ax4 = axes[1, 1]
    ax4.set_title('Node Type Distribution', fontsize=12, fontweight='bold')
    
    node_types = Counter(G.nodes[n]['node_type'] for n in G.nodes())
    colors_nodes = {'MemoryEntry': 'skyblue', 'Entity': 'salmon'}
    
    labels = list(node_types.keys())
    values = list(node_types.values())
    bar_colors = [colors_nodes.get(label, 'gray') for label in labels]
    
    wedges, texts, autotexts = ax4.pie(values, labels=labels, colors=bar_colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Add count labels
    for i, (label, value) in enumerate(zip(labels, values)):
        texts[i].set_text(f'{label}\n({value} nodes)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()


def compare_graphs(data_dir='data/synthetic_demo', num_graphs=3):
    """
    Compare multiple graphs side by side.
    """
    print("=" * 70)
    print(f"COMPARING {num_graphs} GRAPHS")
    print("=" * 70)
    print()
    
    graphs = load_dataset(data_dir, 'train')
    _, graphs_meta = load_metadata(data_dir)
    
    # Prepare comparison table
    print(f"{'Graph ID':<10} {'Nodes':<8} {'Edges':<8} {'Density':<10} {'Connected':<12} {'Sessions':<10}")
    print("-" * 70)
    
    for i in range(min(num_graphs, len(graphs))):
        G = graphs[i]
        meta = graphs_meta[i]
        
        print(f"{i:<10} {G.number_of_nodes():<8} {G.number_of_edges():<8} "
              f"{nx.density(G):<10.6f} {str(nx.is_weakly_connected(G)):<12} "
              f"{meta['params']['num_sessions']:<10}")
    
    print()


def check_connectivity_issues(data_dir='data/synthetic_demo'):
    """
    Diagnose connectivity issues in disconnected graphs.
    """
    print("=" * 70)
    print("CONNECTIVITY DIAGNOSIS")
    print("=" * 70)
    print()
    
    graphs = load_dataset(data_dir, 'train')
    
    disconnected_count = 0
    for i, G in enumerate(graphs):
        if not nx.is_weakly_connected(G):
            disconnected_count += 1
            print(f"Graph {i}: DISCONNECTED")
            
            components = list(nx.weakly_connected_components(G))
            print(f"  Components: {len(components)}")
            print(f"  Component sizes: {sorted([len(c) for c in components], reverse=True)}")
            
            # Check entity distribution
            entity_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Entity']
            print(f"  Total entities: {len(entity_nodes)}")
            
            # Check which components have entities
            for j, comp in enumerate(components):
                entities_in_comp = [n for n in comp if G.nodes[n]['node_type'] == 'Entity']
                print(f"    Component {j}: {len(comp)} nodes, {len(entities_in_comp)} entities")
            
            print()
    
    print(f"Total disconnected graphs: {disconnected_count} / {len(graphs)}")
    
    if disconnected_count > 0:
        print("\n⚠️  RECOMMENDATIONS:")
        print("  1. Increase num_entities (try 50-100 for small graphs)")
        print("  2. Check ensure_connectivity is True in EntityConfig")
        print("  3. Increase mentions_per_event_max to 4-5")
        print("  4. This is less critical for training but should be fixed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Set path
    data_dir = 'data/synthetic_demo'
    
    if not Path(data_dir).exists():
        print(f"❌ Directory not found: {data_dir}")
        print("Run the demo first: python -m src.generation.smsg_pipeline")
        sys.exit(1)
    
    print("🔍 GRAPH INSPECTION TOOL")
    print()
    
    # 1. Analyze output summary
    print("STEP 1: Understanding Generation Output")
    print("=" * 70)
    dataset_meta, graphs_meta = analyze_output_summary(data_dir)
    print()
    
    input("Press Enter to continue to graph inspection...")
    print()
    
    # 2. Inspect single graph
    print("STEP 2: Deep Dive into a Single Graph")
    print("=" * 70)
    graphs = load_dataset(data_dir, 'train')
    G = graphs[0]
    inspect_single_graph(G, graph_id=0)
    print()
    
    input("Press Enter to continue to visualization...")
    print()
    
    # 3. Visualize
    print("STEP 3: Creating Visualization")
    print("=" * 70)
    save_path = 'graph_visualization.png'
    visualize_graph_structure(G, graph_id=0, save_path=save_path)
    print()
    
    # 4. Compare graphs
    print("STEP 4: Comparing Multiple Graphs")
    print("=" * 70)
    compare_graphs(data_dir, num_graphs=min(6, len(graphs)))
    print()
    
    # 5. Check connectivity
    print("STEP 5: Connectivity Diagnosis")
    print("=" * 70)
    check_connectivity_issues(data_dir)
    print()
    
    print("=" * 70)
    print("✅ INSPECTION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the visualization: graph_visualization.png")
    print("  2. If connectivity issues exist, adjust parameters")
    print("  3. Run full generation: python -m src.generation.smsg_pipeline --num_graphs 1000")