import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_graph import CoarseCausalGraph, EventNode, QuerySpec
from refinement_dataset import RefinementTensorDataset, refinement_sample_from_dict
from run_refinement import build_refined_graph


class RefinementGraphEditorTest(unittest.TestCase):
    def test_legacy_query_features_are_ignored_by_dataset(self):
        sample = refinement_sample_from_dict({
            "sample_id": "legacy",
            "node_features": [[0.0] * 10],
            "edge_index": [],
            "edge_features": [],
            "edge_labels": [],
            "edge_strengths": [],
            "query_features": [1.0] * 6,
        })
        item = RefinementTensorDataset([sample])[0]
        self.assertNotIn("query_features", item)
        self.assertNotIn("query_features", sample.to_dict())

    def test_refinement_decoder_returns_acyclic_graph(self):
        events = [EventNode(str(i), str(i), str(i), "doc", i) for i in range(3)]
        coarse = CoarseCausalGraph(QuerySpec("q", "query"), [], events, [])
        candidates = [
            {"source_event_id": "0", "target_event_id": "1"},
            {"source_event_id": "1", "target_event_id": "0"},
            {"source_event_id": "1", "target_event_id": "2"},
            {"source_event_id": "2", "target_event_id": "0"},
        ]
        refined = build_refined_graph(
            coarse,
            candidates,
            keep_probs=[0.9, 0.8, 0.7, 0.6],
            strength_predictions=[0.9, 0.8, 0.7, 0.6],
            keep_threshold=0.5,
            topology_mode="temporal-dag",
        )
        self.assertEqual(len(refined.edges), 2)
        self.assertEqual(refined.metadata["refinement_topology"]["reciprocal_pruned_count"], 1)
        self.assertEqual(refined.metadata["refinement_topology"]["cycle_pruned_count"], 1)


if __name__ == "__main__":
    unittest.main()
