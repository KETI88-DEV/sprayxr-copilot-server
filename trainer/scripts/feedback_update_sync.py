"""Collect pending feedback and merge reachable peer receipts before an update.

Run: python -m trainer.scripts.feedback_update_sync --main-url URL [--peer-url URL] [--merge]
Credentials: COPILOT_ADMIN_USER/PASSWORD; optional COPILOT_PEER_ADMIN_USER/PASSWORD.
No opinion is marked done by this command. Implement and verify it, then update its status.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import urlsplit

import httpx


def login(url: str, prefix: str) -> httpx.Client:
    parts = urlsplit(url)
    if parts.scheme not in {'http', 'https'} or not parts.netloc or parts.username or parts.password or parts.path not in {'', '/'}:
        raise ValueError('Use a server origin URL without credentials or a path')
    origin = url.rstrip('/')
    user = os.environ.get(prefix+'_USER', os.environ.get('COPILOT_ADMIN_USER', 'admin'))
    password = os.environ.get(prefix+'_PASSWORD', os.environ.get('COPILOT_ADMIN_PASSWORD'))
    if not password:
        raise ValueError('Administrator password environment variable is required')
    client = httpx.Client(base_url=origin, timeout=20, headers={'Origin': origin})
    try:
        response = client.post('/admin/api/manage/login', json={'username': user, 'password': password})
        response.raise_for_status()
        client.headers['X-DB-CSRF'] = response.json()['csrf_token']
        return client
    except BaseException:
        client.close()
        raise


def transfer(source: httpx.Client, target: httpx.Client) -> dict:
    offset = 0
    counts = dict(inserted=0, updated=0, unchanged=0, conflict=0)
    while True:
        response = source.get('/admin/api/feedback/sync-bundle', params={'offset': offset})
        response.raise_for_status()
        bundle = response.json()
        imported = target.post('/admin/api/manage/feedback-sync', json=bundle)
        imported.raise_for_status()
        result = imported.json()
        for key in counts:
            counts[key] += result[key]
        following = bundle.get('next_offset')
        if following is None:
            return counts
        if not isinstance(following, int) or following <= offset:
            raise ValueError('Peer returned invalid pagination')
        offset = following


def pending(client: httpx.Client):
    result=[]
    for status in ['new', 'reviewing']:
        page=1
        while True:
            response=client.get('/api/feedback/list',params={'status':status,'page':page})
            response.raise_for_status()
            data=response.json()
            result.extend(data['items'])
            if page*20 >= data['total']:
                break
            page+=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-url',required=True)
    parser.add_argument('--peer-url')
    parser.add_argument('--merge',action='store_true',help='Import both directions using UUID and review timestamps')
    args=parser.parse_args()
    result={'main':{},'peer':{'status':'not_requested'}}
    with login(args.main_url, 'COPILOT_ADMIN') as primary:
        if args.peer_url:
            try:
                with login(args.peer_url,'COPILOT_PEER_ADMIN') as peer:
                    if args.merge:
                        result['peer_to_main']=transfer(peer,primary)
                        result['main_to_peer']=transfer(primary,peer)
                    result['peer']={'status':'checked','pending':pending(peer)}
            except (httpx.HTTPError, ValueError, KeyError):
                # Report the gap rather than treating an unreachable/old peer as synchronized.
                result['peer']={'status':'unavailable','message':'Peer unreachable or feedback API upgrade/authentication required'}
        result['main']={'status':'checked','pending':pending(primary)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2 if result['peer']['status']=='unavailable' else 0


if __name__=='__main__':
    sys.exit(main())
