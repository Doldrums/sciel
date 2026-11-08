#!/usr/bin/env python3
"""Logging HTTP(S) proxy for research_tools.network: the only way out of the internal network tools run on.

    egress_proxy.py --log-dir DIR [--port 3128] [--allow HOST_SUFFIX ...]

A tool container gets HTTP(S)_PROXY = http://<container name>:x@<proxy>; the name is the proxy user, so every request
is written to DIR/<container name>.jsonl (time, method, host, port, url for plain HTTP, status, bytes each way,
seconds), and to DIR/all.jsonl. CONNECT tunnels HTTPS, so for those the log has the host, not the path. Requests
without a sciel container name get 407; with --allow, hosts outside the listed suffixes get 403.

Run it in a container on the default bridge (for the internet) joined to the internal network under the proxy's name:
    docker network create --internal sciel-egress
    docker run -d --name sciel-proxy --restart unless-stopped -v $PWD/scripts/egress_proxy.py:/proxy.py:ro \\
        -v /ephemeral/egress_logs:/logs IMAGE python /proxy.py --log-dir /logs
    docker network connect --alias sciel-proxy sciel-egress sciel-proxy
and set research_tools.network = {docker_network: sciel-egress, proxy: http://sciel-proxy:3128,
log_dir: /ephemeral/egress_logs}.
"""
import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

NAME = re.compile(r'sciel-(method|eval)-[0-9a-f]{16,64}\Z')
HEAD_LIMIT = 65536
IDLE_SECONDS = 600


def _user(headers):
    raw = headers.get('proxy-authorization', '')
    if not raw.lower().startswith('basic '):
        return None
    try:
        user = base64.b64decode(raw[6:]).decode().split(':', 1)[0]
    except (ValueError, UnicodeError):
        return None
    return user if NAME.fullmatch(user) else None


class Proxy:
    def __init__(self, log_dir, allow):
        self.log_dir, self.allow = Path(log_dir), tuple(a.lower().lstrip('.') for a in allow)

    def log(self, entry):
        line = json.dumps(entry, sort_keys=True) + '\n'
        for name in (entry.get('container') or 'unknown', 'all'):
            with open(self.log_dir / (name + '.jsonl'), 'a') as f:
                f.write(line)

    def allowed(self, host):
        host = host.lower()
        return not self.allow or any(host == a or host.endswith('.' + a) for a in self.allow)

    async def pipe(self, reader, writer, count):
        try:
            while True:
                data = await asyncio.wait_for(reader.read(65536), IDLE_SECONDS)
                if not data:
                    break
                count[0] += len(data)
                writer.write(data)
                await writer.drain()
        except (OSError, asyncio.TimeoutError):
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass

    async def handle(self, reader, writer):
        begun = time.time()
        entry = dict(time=round(begun, 3), container=None, method=None, host=None, port=None, url=None,
                     status=None, bytes_up=0, bytes_down=0)
        try:
            head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 30)
            if len(head) > HEAD_LIMIT:
                raise ValueError('header too large')
            lines = head.decode('latin-1').split('\r\n')
            method, target, version = lines[0].split(' ', 2)
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    k, v = line.split(':', 1)
                    headers[k.strip().lower()] = v.strip()
            entry.update(method=method, container=_user(headers))
            if method == 'CONNECT':
                host, _, port = target.rpartition(':')
                port = int(port or 443)
            else:
                url = urlsplit(target)
                host, port = url.hostname or '', url.port or 80
                entry['url'] = target[:500]
            entry.update(host=host, port=port)
            if entry['container'] is None:
                entry['status'] = 407
                writer.write(b'HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic\r\n'
                             b'Content-Length: 0\r\nConnection: close\r\n\r\n')
                return
            if not self.allowed(host):
                entry['status'] = 403
                writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                return
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, port), 30)
            up, down = [0], [0]
            if method == 'CONNECT':
                entry['status'] = 200
                writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                await writer.drain()
            else:
                path = url.path or '/'
                if url.query:
                    path += '?' + url.query
                kept = [l for l in lines[1:] if l and not l.lower().startswith(('proxy-', 'connection:'))]
                request = '\r\n'.join([f'{method} {path} {version}', *kept, 'Connection: close', '', ''])
                up_writer.write(request.encode('latin-1'))
                await up_writer.drain()
                up[0] += len(request)
                first = await asyncio.wait_for(up_reader.readline(), 60)
                down[0] += len(first)
                writer.write(first)
                parts = first.split()
                entry['status'] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            await asyncio.gather(self.pipe(reader, up_writer, up), self.pipe(up_reader, writer, down))
            entry.update(bytes_up=up[0], bytes_down=down[0])
        except (OSError, ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError,
                asyncio.LimitOverrunError) as exc:
            entry['error'] = type(exc).__name__ + ': ' + str(exc)[:200]
            if entry['status'] is None:
                entry['status'] = 502
                try:
                    writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                except OSError:
                    pass
        finally:
            entry['seconds'] = round(time.time() - begun, 3)
            try:
                writer.close()
            except OSError:
                pass
            self.log(entry)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--port', type=int, default=3128)
    parser.add_argument('--allow', nargs='*', default=[])
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    proxy = Proxy(args.log_dir, args.allow)
    server = await asyncio.start_server(proxy.handle, '0.0.0.0', args.port, limit=HEAD_LIMIT * 2)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
