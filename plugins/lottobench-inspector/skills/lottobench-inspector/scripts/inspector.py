"""Dependency-free reader for pinned LottoBench evidence. No network or writes."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parents[1] / 'references' / 'snapshot.json'
GAMES = ('euromillions', 'nl-lotto')
METRICS = {
    'mean_roi_alpha_pp': 'modeled percentage-point difference from uniform_random',
    'mean_expected_roi_pct': 'modeled expected ROI percent',
    'mean_pair_coverage_pct': 'pair coverage percent',
    'mean_hit_recall_pct': 'observed holdout recall percent; not a predictive edge',
    'mean_unpopularity_lift_x': 'modeled conditional payout multiplier',
}


def load_snapshot(path=SNAPSHOT):
    snapshot = json.loads(Path(path).read_text(encoding='utf-8'))
    if snapshot['schema_version'] != 1:
        raise ValueError('Unsupported snapshot schema')
    if snapshot['repository'] != 'kugguk2022/lotteries-init-at-your-service':
        raise ValueError('Unexpected source repository')
    ref = snapshot['ref']
    if len(ref) != 40 or any(c not in '0123456789abcdef' for c in ref):
        raise ValueError('Snapshot must be pinned to a full commit SHA')
    sources = {}
    for source in snapshot['sources']:
        path = source['path']
        if path in sources:
            raise ValueError(f'Duplicate source: {path}')
        payload = source['content'].encode('utf-8')
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        if blob != source['blob_sha'] or hashlib.sha256(payload).hexdigest() != source['sha256']:
            raise ValueError(f'Source integrity mismatch: {path}')
        sources[path] = source
    return snapshot, sources


def evidence(snapshot, *sources):
    return {
        'repository': snapshot['repository'], 'source_commit': snapshot['ref'],
        'captured_utc': snapshot['captured_utc'], 'mode': 'pinned_snapshot',
        'integrity': 'Git blob SHA and SHA-256 verified; upstream truth not independently audited',
        'sources': [
            {'path': s['path'], 'sha256': s['sha256'],
             'url': f"https://github.com/{snapshot['repository']}/blob/{snapshot['ref']}/{s['path']}"}
            for s in sources
        ],
    }


def provider_catalog(snapshot, sources):
    source = sources['lotteries_core/registry.py']
    # Inspect literals without importing or executing code from a repository.
    tree = ast.parse(source['content'])
    spec_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ProviderSpec')
    defaults = {node.target.id: ast.literal_eval(node.value) for node in spec_class.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in ('version', 'optional', 'install_extra', 'ablation_of')}
    if set(defaults) != {'version', 'optional', 'install_extra', 'ablation_of'}:
        raise ValueError('Registry defaults changed; review the source')
    value = next(node.value for node in tree.body if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) and node.target.id == 'PROVIDERS')
    providers = []
    for key, node in zip(value.keys, value.values):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'ProviderSpec':
            raise ValueError('Registry structure changed; review the source')
        args = [ast.literal_eval(item) for item in node.args]
        if len(args) != 3 or ast.literal_eval(key) != args[0]:
            raise ValueError('Registry provider identity mismatch')
        row = dict(zip(('name', 'summary', 'implementation'), args))
        row.update(defaults)
        for item in node.keywords:
            if item.arg not in ('version', 'optional', 'install_extra', 'ablation_of'):
                raise ValueError('Registry keyword changed; review the source')
            row[item.arg] = ast.literal_eval(item.value)
        row['runtime_available'] = None
        providers.append(row)
    return {'provider_count': len(providers), 'providers': providers,
            'availability_note': 'Registry identities only; algorithms were not instantiated.',
            'baseline': 'uniform_random', 'evidence': evidence(snapshot, source)}


def profile(snapshot, sources, game):
    if game not in GAMES:
        raise ValueError(f'Unknown game: {game}')
    base = f'publishing/huggingface-space/data/profiles/{game}/'
    manifest_source = sources[base + 'manifest.json']
    board_source = sources[base + 'leaderboard.csv']
    prospective_source = sources[base + 'prospective.csv']
    manifest = json.loads(manifest_source['content'])
    if manifest['profile_key'] != game or manifest['game']['name'] != game:
        raise ValueError('Game identity mismatch')
    if manifest['data_kind'] != 'observed_public_history' or manifest['evaluation']['protocol'] != 'forward_only':
        raise ValueError('Unsupported data kind or evaluation protocol')
    if manifest['evaluation']['house_reference'] != 'uniform_random':
        raise ValueError('Unexpected baseline')
    rows = list(csv.DictReader(io.StringIO(board_source['content'])))
    agents = set()
    for row in rows:
        if row['agent'] in agents:
            raise ValueError('Duplicate leaderboard agent')
        agents.add(row['agent'])
        for key in METRICS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Non-finite benchmark metric')
        row['contests'] = int(row['contests'])
        if row['contests'] != manifest['evaluation']['holdout_contests']:
            raise ValueError('Leaderboard holdout differs from manifest')
    if agents != set(manifest['evaluation']['providers']) | {manifest['evaluation']['coordinator']}:
        raise ValueError('Leaderboard agent identities differ from manifest')
    null = next(row for row in rows if row['agent'] == 'uniform_random')
    for row in rows:
        expected = row['mean_expected_roi_pct'] - null['mean_expected_roi_pct']
        if not math.isclose(row['mean_roi_alpha_pp'], expected, abs_tol=1e-8):
            raise ValueError('ROI alpha differs from the stated uniform baseline')
    prospective = list(csv.DictReader(io.StringIO(prospective_source['content'])))
    statuses = {}
    for row in prospective:
        status = row['score_status']
        statuses[status] = statuses.get(status, 0) + 1
    return manifest, rows, statuses, (manifest_source, board_source, prospective_source)


def dataset_status(snapshot, sources, game, today=None):
    manifest, _, statuses, src = profile(snapshot, sources, game)
    history = manifest['history']
    cutoff = date.fromisoformat(history['last_draw'])
    today = today or datetime.now(timezone.utc).date()
    if cutoff > today:
        raise ValueError('History cutoff is in the future')
    age = (today - cutoff).days
    return {'game': game, 'as_of_date': today.isoformat(), 'history': history,
            'history_age_days': age, 'freshness': 'older_than_7_days' if age > 7 else 'within_7_days',
            'freshness_note': 'Age of the bundled profile; not a check for missing official draws.',
            'data_kind': manifest['data_kind'], 'source': manifest['source'],
            'evaluation': manifest['evaluation'], 'prospective_ticket_status_counts': statuses,
            'realized_roi': None,
            'realized_roi_status': 'unavailable_in_inspected_profile; no validated cost/payout settlement supplied',
            'claims_boundary': manifest['claims_boundary'], 'evidence': evidence(snapshot, *src)}


def leaderboard(snapshot, sources, game, metric='mean_roi_alpha_pp'):
    if metric not in METRICS:
        raise ValueError(f'Unknown metric: {metric}')
    manifest, rows, _, src = profile(snapshot, sources, game)
    ordered = sorted(rows, key=lambda r: (-r[metric], r['agent']))
    return {'game': game, 'metric': metric, 'unit': METRICS[metric], 'direction': 'higher',
            'evaluation': manifest['evaluation'], 'history_cutoff': manifest['history']['last_draw'],
            'rows': [{'agent': r['agent'], 'label': r['agent_label'], 'value': r[metric]} for r in ordered],
            'claims_boundary': manifest['claims_boundary'], 'evidence': evidence(snapshot, *src)}


def compare(snapshot, sources, game, a, b, metric='mean_roi_alpha_pp'):
    board = leaderboard(snapshot, sources, game, metric)
    values = {row['agent']: row['value'] for row in board['rows']}
    if a not in values or b not in values:
        raise ValueError(f'Agent absent from this cohort; choose from {sorted(values)}')
    return {k: v for k, v in board.items() if k != 'rows'} | {
        'a': {'agent': a, 'value': values[a]}, 'b': {'agent': b, 'value': values[b]},
        'a_minus_b': values[a] - values[b], 'comparison_scope': 'same stored game/profile/cohort',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('summary', 'providers', 'status', 'leaderboard', 'compare'))
    parser.add_argument('--game', choices=GAMES, default='euromillions')
    parser.add_argument('--metric', choices=tuple(METRICS), default='mean_roi_alpha_pp')
    parser.add_argument('--a')
    parser.add_argument('--b', default='uniform_random')
    args = parser.parse_args()
    try:
        snapshot, sources = load_snapshot()
        if args.command == 'summary':
            result = {game: {'status': dataset_status(snapshot, sources, game),
                             'leaderboard': leaderboard(snapshot, sources, game)} for game in GAMES}
        elif args.command == 'providers':
            result = provider_catalog(snapshot, sources)
        elif args.command == 'status':
            result = dataset_status(snapshot, sources, args.game)
        elif args.command == 'leaderboard':
            result = leaderboard(snapshot, sources, args.game, args.metric)
        else:
            if not args.a:
                parser.error('compare requires --a')
            result = compare(snapshot, sources, args.game, args.a, args.b, args.metric)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, StopIteration, SyntaxError) as exc:
        parser.exit(2, f'Evidence error: {exc}\n')


if __name__ == '__main__':
    main()
