"""
Generation package for Memory-R1 synthetic graph generation.

This package provides the SMSG (Schema-Mimetic Synthetic Generation) pipeline
for creating synthetic graphs with temporal, entity, and causal structures.
"""

# Phase 1: Temporal Backbone
from .backbone_generator import TemporalBackboneGenerator, BackboneConfig

# Phase 2: Entity Injection
from .entity_injection import EntityInjector, EntityConfig

# Phase 3: Causal Wiring
from .causal_wiring import CausalWiring, CausalConfig

# Complete Pipeline
from .smsg_pipeline import SMSGPipeline, SMSGConfig, load_dataset, load_metadata

__all__ = [
    # Phase 1
    'TemporalBackboneGenerator',
    'BackboneConfig',
    # Phase 2
    'EntityInjector',
    'EntityConfig',
    # Phase 3
    'CausalWiring',
    'CausalConfig',
    # Pipeline
    'SMSGPipeline',
    'SMSGConfig',
    'load_dataset',
    'load_metadata',
]
