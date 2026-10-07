"""Local stdio MCP adapter, pinned to the 2025-06-18 handshake protocol."""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
from inspector import GAMES, METRICS, compare, dataset_status, leaderboard, load_snapshot, provider_catalog

VERSION = '0.1.0'
PROTOCOL = '2025-06-18'
GAME = {'type': 'string', 'enum': list(GAMES)}
METRIC = {'type': 'string', 'enum': list(METRICS), 'default': 'mean_roi_alpha_pp'}


def tool(name, description, properties, required=()):
    return {'name': name, 'description': description,
            'inputSchema': {'type': 'object', 'properties': properties,
                            'required': list(required), 'additionalProperties': False},
            'annotations': {'readOnlyHint': True, 'destructiveHint': False,
                            'idempotentHint': True, 'openWorldHint': False}}


TOOLS = [
    tool('lottobench_providers', 'List pinned registry identities, versions, families and ablations. Does not execute algorithms.', {}),
    tool('lottobench_dataset_status', 'Inspect game-profile provenance, history age and pending evidence; realized ROI is unavailable without settlements.', {'game': GAME}, ('game',)),
    tool('lottobench_benchmark_results', 'Read a stored forward-only leaderboard. ROI is modeled, never settled cash profit.', {'game': GAME, 'metric': METRIC}, ('game',)),
    tool('lottobench_compare', 'Compare two agents in the same stored cohort, with metric units and source hashes.', {'game': GAME, 'a': {'type': 'string'}, 'b': {'type': 'string'}, 'metric': METRIC}, ('game', 'a', 'b')),
]


class RpcError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


class Server:
    def __init__(self):
        self.initializing = False
        self.ready = False

    def handle(self, message):
        if not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or not isinstance(message.get('method'), str):
            return self.error(None, -32600, 'Invalid JSON-RPC request')
        request_id = message.get('id')
        if 'id' in message and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
            return self.error(None, -32600, 'Invalid request ID')
        method = message['method']
        # Notifications never receive a response, including unsupported ones.
        if 'id' not in message:
            if method == 'notifications/initialized' and self.initializing:
                self.ready = True
            return None
        try:
            params = message.get('params', {})
            if not isinstance(params, dict):
                raise RpcError(-32602, 'Parameters must be an object')
            if method == 'initialize':
                if self.initializing:
                    raise RpcError(-32600, 'Already initialized')
                if not isinstance(params.get('protocolVersion'), str) or not isinstance(params.get('capabilities'), dict) or not isinstance(params.get('clientInfo'), dict):
                    raise RpcError(-32602, 'Missing initialization fields')
                self.initializing = True
                result = {'protocolVersion': PROTOCOL, 'capabilities': {'tools': {'listChanged': False}},
                          'serverInfo': {'name': 'lottobench-inspector', 'version': VERSION},
                          'instructions': 'Pinned research snapshots only; modeled ROI is not realized profit.'}
            elif method == 'ping':
                result = {}
            elif not self.ready:
                raise RpcError(-32002, 'Initialize and send notifications/initialized first')
            elif method == 'tools/list':
                if params.get('cursor') is not None:
                    raise RpcError(-32602, 'This server has no pagination cursor')
                result = {'tools': TOOLS}
            elif method == 'tools/call':
                spec = next((t for t in TOOLS if t['name'] == params.get('name')), None)
                if spec is None:
                    raise RpcError(-32602, 'Unknown tool')
                args = params.get('arguments', {})
                schema = spec['inputSchema']
                if not isinstance(args, dict) or set(args) - set(schema['properties']) or set(schema['required']) - set(args):
                    raise RpcError(-32602, 'Invalid tool arguments')
                for key, value in args.items():
                    rule = schema['properties'][key]
                    if not isinstance(value, str) or ('enum' in rule and value not in rule['enum']):
                        raise RpcError(-32602, f'Invalid argument: {key}')
                try:
                    snapshot, sources = load_snapshot()
                    name = spec['name']
                    if name == 'lottobench_providers':
                        data = provider_catalog(snapshot, sources)
                    elif name == 'lottobench_dataset_status':
                        data = dataset_status(snapshot, sources, args['game'])
                    elif name == 'lottobench_benchmark_results':
                        data = leaderboard(snapshot, sources, args['game'], args.get('metric', 'mean_roi_alpha_pp'))
                    else:
                        data = compare(snapshot, sources, args['game'], args['a'], args['b'], args.get('metric', 'mean_roi_alpha_pp'))
                    result = {'content': [{'type': 'text', 'text': json.dumps(data, allow_nan=False)}],
                              'structuredContent': data, 'isError': False}
                except (ValueError, KeyError, OSError, StopIteration, SyntaxError) as exc:
                    result = {'content': [{'type': 'text', 'text': f'Evidence error: {exc}'}], 'isError': True}
            else:
                raise RpcError(-32601, 'Method not found')
            return {'jsonrpc': '2.0', 'id': request_id, 'result': result}
        except RpcError as exc:
            return self.error(request_id, exc.code, exc.message)

    @staticmethod
    def error(request_id, code, message):
        return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


def main():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    server = Server()
    while line := sys.stdin.readline(1024 * 1024 + 1):
        if len(line) > 1024 * 1024:
            response = server.error(None, -32600, 'Request exceeds 1 MiB')
            print(json.dumps(response), flush=True)
            break
        try:
            message = json.loads(line)
            response = server.handle(message)
        except (json.JSONDecodeError, UnicodeError):
            response = server.error(None, -32700, 'Parse error')
        if response is not None:
            print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
