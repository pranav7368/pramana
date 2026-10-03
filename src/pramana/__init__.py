"""PRAMANA — Pipeline for Retrieval-Augmented Multilingual Assurance
via NLI-grounded Auto-correction.

A black-box assurance layer for multilingual enterprise RAG systems: it
decomposes generated answers into atomic claims, verifies each against retrieved
evidence, produces a calibrated confidence score, and applies evidence-guided
correction or abstention.

See docs/04_SYSTEM_ARCHITECTURE.md for the design.
"""

__version__ = "0.1.0"
