"""Fetch public web pages using a validated, pinned destination IP."""
import ipaddress
import socket
import time
from urllib.parse import urlparse, urljoin

import urllib3

MAX_BYTES = 2 * 1024 * 1024


def public_destination(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Provide a valid HTTP or HTTPS URL.')
    if parsed.username or parsed.password:
        raise ValueError('URL credentials are not supported.')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if port not in (80, 443):
        raise ValueError('Only standard web ports are supported.')
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    ips = {item[4][0] for item in addresses}
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError('Private and reserved network destinations are blocked.')
    return parsed, port, sorted(ips)[0]


def fetch_public(url, headers=None):
    deadline = time.monotonic() + 25
    for _ in range(4):
        parsed, port, ip = public_destination(url)
        pool_class = urllib3.HTTPSConnectionPool if parsed.scheme == 'https' else urllib3.HTTPConnectionPool
        tls = {'server_hostname': parsed.hostname, 'assert_hostname': parsed.hostname} if parsed.scheme == 'https' else {}
        pool = pool_class(ip, port=port, **tls)
        response = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('Article fetch timed out.')
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query
            request_headers = dict(headers or {})
            request_headers['Host'] = parsed.netloc
            response = pool.request('GET', path, headers=request_headers,
                                    redirect=False, retries=False, preload_content=False,
                                    timeout=urllib3.Timeout(connect=min(5, remaining), read=min(5, remaining)))
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if not location:
                    raise ValueError('Invalid redirect.')
                url = urljoin(url, location)
                continue
            if response.status >= 400:
                raise ValueError(f'Article server returned HTTP {response.status}.')
            content_type = response.headers.get('Content-Type', '').lower()
            if not any(kind in content_type for kind in ('text/html', 'application/xhtml+xml', 'application/json')):
                raise ValueError('URL did not return an HTML page or JSON.')
            chunks, size = [], 0
            for chunk in response.stream(65536, decode_content=True):
                size += len(chunk)
                if size > MAX_BYTES or time.monotonic() > deadline:
                    raise ValueError('Article response is too large or too slow.')
                chunks.append(chunk)
            return b''.join(chunks).decode('utf-8', errors='replace')
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError('Too many redirects.')
