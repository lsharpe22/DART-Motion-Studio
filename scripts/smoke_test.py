#!/usr/bin/env python3
"""Authenticated generation smoke test. No external Python packages required."""
import argparse
import json
import time
import urllib.error
import urllib.request

def req(url, endpoint, key, payload=None):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['X-API-Key'] = key
    request = urllib.request.Request(url.rstrip('/') + endpoint, data=data, headers=headers, method='POST' if payload is not None else 'GET')
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8012')
    p.add_argument('--api-key', required=True)
    p.add_argument('--prompt', default='walk forward')
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--continue-prompt', default='')
    a = p.parse_args()
    print('Health:', req(a.url, '/health', a.api_key))
    print('Engine:', req(a.url, '/engine', a.api_key))
    def generate(endpoint, payload):
        t0 = time.perf_counter()
        r = req(a.url, endpoint, a.api_key, payload)
        assert r.get('status') == 'success', r.get('status')
        assert r.get('session_id'), 'Missing session ID'
        assert r.get('frames', 0) > 0, 'No generated frames'
        assert r.get('joint_count') == 22, 'Expected 22 joints'
        assert len(r['joints']) == r['frames'], 'Frame count mismatch'
        print(endpoint, {'frames': r['frames'], 'joint_count': r['joint_count'], 'fps': r['fps'], 'seconds': round(time.perf_counter()-t0, 3), 'session_present': True})
        return r
    first = generate('/generate', {'prompt': a.prompt, 'repetitions': a.repetitions})
    if a.continue_prompt:
        generate('/continue', {'prompt': a.continue_prompt, 'repetitions': a.repetitions, 'session_id': first['session_id']})
    print('PASS')
if __name__=='__main__':
    try:
        main()
    except urllib.error.HTTPError as e:
        print('HTTP error:', e.code, e.read().decode('utf-8', errors='replace')[:500])
        raise SystemExit(1)
