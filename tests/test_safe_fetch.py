import unittest
from unittest.mock import patch, MagicMock
from backend.src.safe_fetch import public_destination, fetch_public, MAX_BYTES


def dns(ip):
    return [(2, 1, 6, '', (ip, 443))]


class PublicFetchTests(unittest.TestCase):
    def test_private_reserved_and_mixed_dns_blocked(self):
        for ip in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '192.168.0.1'):
            with self.subTest(ip=ip), patch('socket.getaddrinfo', return_value=dns(ip)):
                with self.assertRaises(ValueError):
                    public_destination('https://example.com')
        with patch('socket.getaddrinfo', return_value=dns('8.8.8.8') + dns('127.0.0.1')):
            with self.assertRaises(ValueError):
                public_destination('https://example.com')

    def test_bad_scheme_credentials_and_ports(self):
        for url in ('file:///etc/passwd', 'http://u:p@example.com', 'https://example.com:5000'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                public_destination(url)

    @patch('backend.src.safe_fetch.urllib3.HTTPSConnectionPool')
    @patch('socket.getaddrinfo', return_value=dns('8.8.8.8'))
    def test_pins_connection_and_preserves_tls_hostname(self, resolve, pool):
        response = MagicMock(status=200, headers={'Content-Type': 'text/html'})
        response.stream.return_value = [b'<html>article</html>']
        pool.return_value.request.return_value = response
        self.assertIn('article', fetch_public('https://example.com/story'))
        pool.assert_called_once_with('8.8.8.8', port=443, server_hostname='example.com', assert_hostname='example.com')
        self.assertFalse(pool.return_value.request.call_args.kwargs['redirect'])

    @patch('backend.src.safe_fetch.urllib3.HTTPSConnectionPool')
    @patch('socket.getaddrinfo', side_effect=[dns('8.8.8.8'), dns('127.0.0.1')])
    def test_redirect_private_destination_blocked(self, resolve, pool):
        pool.return_value.request.return_value = MagicMock(status=302, headers={'Location': 'https://internal.example/'})
        with self.assertRaises(ValueError):
            fetch_public('https://example.com')
        self.assertEqual(pool.call_count, 1)

    @patch('backend.src.safe_fetch.urllib3.HTTPConnectionPool')
    @patch('socket.getaddrinfo', return_value=dns('8.8.8.8'))
    def test_body_size_limit(self, resolve, pool):
        response = MagicMock(status=200, headers={'Content-Type': 'text/html'})
        response.stream.return_value = [b'x' * (MAX_BYTES + 1)]
        pool.return_value.request.return_value = response
        with self.assertRaises(ValueError):
            fetch_public('http://example.com')


if __name__ == '__main__':
    unittest.main()
