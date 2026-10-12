"""Integrity, evidence-boundary and real subprocess wire checks."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
SCRIPTS = PLUGIN / 'skills' / 'lottobench-inspector' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import inspector  # noqa: E402
from mcp_server import Server  # noqa: E402


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.snapshot, self.sources = inspector.load_snapshot()

    def altered(self, suffix, change):
        snapshot = copy.deepcopy(self.snapshot)
        source = next(s for s in snapshot['sources'] if s['path'].endswith(suffix))
        source['content'] = change(source['content'])
        payload = source['content'].encode()
        source['blob_sha'] = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        source['sha256'] = hashlib.sha256(payload).hexdigest()
        return snapshot, {s['path']: s for s in snapshot['sources']}

    def test_registry_preserves_named_models_and_ablation(self):
        providers = {p['name']: p for p in inspector.provider_catalog(self.snapshot, self.sources)['providers']}
        self.assertEqual(len(providers), 12)
        self.assertEqual(providers['gingerm']['implementation'], 'cooccurrence_level_set')
        self.assertEqual(providers['parallax_guard_ablation']['ablation_of'], 'parallax')
        self.assertEqual(providers['sequence_transformer']['install_extra'], 'transformer')
        self.assertTrue(all(p['runtime_available'] is None for p in providers.values()))

    def test_realized_roi_is_unknown_and_age_uses_history(self):
        for game, age in [('euromillions', 40), ('nl-lotto', 46)]:
            result = inspector.dataset_status(self.snapshot, self.sources, game, date(2026, 10, 7))
            self.assertEqual(result['history_age_days'], age)
            self.assertIsNone(result['realized_roi'])
            self.assertEqual(result['prospective_ticket_status_counts'], {'PENDING': 84})

    def test_comparison_has_percentage_point_units_and_pinned_provenance(self):
        result = inspector.compare(self.snapshot, self.sources, 'nl-lotto', 'gingerm', 'uniform_random')
        self.assertAlmostEqual(result['a_minus_b'], 0.2629550918448094)
        self.assertIn('percentage-point', result['unit'])
        self.assertEqual(result['evaluation']['budget_per_agent'], 12)
        self.assertEqual(result['evidence']['source_commit'], '1de1ce1b40093c761634ba49a5967f7794b4bc2e')

    def test_alternate_metric_changes_order(self):
        result = inspector.leaderboard(self.snapshot, self.sources, 'euromillions', 'mean_pair_coverage_pct')
        self.assertIn(result['rows'][0]['agent'], ('parallax', 'coordinated_aggregation'))

    def test_content_tampering_is_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['sources'][0]['content'] += '\n# tampered\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            path.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                inspector.load_snapshot(path)

    def test_different_holdout_is_rejected(self):
        snapshot, sources = self.altered('euromillions/manifest.json', lambda s: s.replace('"holdout_contests": 12', '"holdout_contests": 13'))
        with self.assertRaisesRegex(ValueError, 'holdout'):
            inspector.leaderboard(snapshot, sources, 'euromillions')

    def test_roi_alpha_inconsistency_is_rejected(self):
        snapshot, sources = self.altered('nl-lotto/leaderboard.csv', lambda s: s.replace('0.39439798476793947', '99.39439798476793947'))
        with self.assertRaisesRegex(ValueError, 'ROI alpha'):
            inspector.leaderboard(snapshot, sources, 'nl-lotto')

    def test_missing_cohort_provider_is_not_silently_invented(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            inspector.compare(self.snapshot, self.sources, 'nl-lotto', 'sequence_transformer', 'uniform_random')

    def test_unknown_game_and_metric_are_rejected(self):
        with self.assertRaises(ValueError):
            inspector.dataset_status(self.snapshot, self.sources, '../ledger')
        with self.assertRaises(ValueError):
            inspector.leaderboard(self.snapshot, self.sources, 'nl-lotto', 'profit')


class WireTests(unittest.TestCase):
    def test_actual_stdio_process_handshake_and_all_four_tools(self):
        messages = [
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'test-client', 'version': '1'}}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        ]
        args = [
            ('lottobench_providers', {}),
            ('lottobench_dataset_status', {'game': 'nl-lotto'}),
            ('lottobench_benchmark_results', {'game': 'euromillions'}),
            ('lottobench_compare', {'game': 'nl-lotto', 'a': 'gingerm', 'b': 'uniform_random'}),
        ]
        for index, (name, arguments) in enumerate(args, 3):
            messages.append({'jsonrpc': '2.0', 'id': index, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        proc = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'mcp_server.py')],
                              input='\n'.join(json.dumps(m) for m in messages) + '\n',
                              capture_output=True, text=True, timeout=10, cwd=tempfile.gettempdir())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, '')
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([r['id'] for r in responses], [1, 2, 3, 4, 5, 6])
        self.assertEqual(responses[0]['result']['protocolVersion'], '2025-06-18')
        self.assertEqual(len(responses[1]['result']['tools']), 4)
        for response in responses[2:]:
            self.assertFalse(response['result']['isError'])
            self.assertIn('evidence', response['result']['structuredContent'])

    def test_tools_require_initialization(self):
        result = Server().handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
        self.assertEqual(result['error']['code'], -32002)

    def test_parse_error_does_not_corrupt_stdout(self):
        proc = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'mcp_server.py')], input='{bad\n', capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(proc.stdout)['error']['code'], -32700)

    def test_tool_cannot_read_arbitrary_path(self):
        server = Server()
        server.ready = True
        result = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {
            'name': 'lottobench_dataset_status', 'arguments': {'game': 'nl-lotto', 'path': '/etc/passwd'}}})
        self.assertEqual(result['error']['code'], -32602)

    def test_manifest_launch_path_resolves(self):
        config = json.loads((PLUGIN / 'mcp.json').read_text())['mcpServers']['lottobench']
        self.assertEqual(config['type'], 'stdio')
        target = Path(config['args'][1].replace('${PLUGIN_ROOT}', str(PLUGIN)))
        self.assertTrue(target.is_file())


if __name__ == '__main__':
    unittest.main()
