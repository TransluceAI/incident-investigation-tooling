"""Inert URL decoding: cases that used to be missed without a recorded gap."""
import base64
import unittest
from urllib.parse import quote

from arquivo_judge.decode import decode_url

# Encodes to text containing "+", which query decoding turns into a space.
PLUS = base64.b64encode(b'hello world, this is a test >>> ok').decode()


def kinds(url, **kwargs):
    views, gaps = decode_url(url, **kwargs)
    return [(v['kind'], v['source']) for v in views], gaps


class DecodeTests(unittest.TestCase):
    def test_base64_with_plus_is_decoded(self):
        self.assertIn('+', PLUS)
        views, _ = decode_url('https://a.example/x?q=' + PLUS)
        self.assertIn('hello world, this is a test >>> ok', [v['text'] for v in views if v['kind'] == 'DECODED_B64'])

    def test_code_named_parameter_is_also_tried_as_base64(self):
        views, _ = kinds('https://a.example/x?code=' + PLUS)
        self.assertIn(('PROGRAM', 'URL.query.code:percent-decode:1'), views)
        self.assertIn(('DECODED_B64', 'URL.query.code:base64:1'), views)

    def test_fragment_parameters_and_urls_are_decoded(self):
        views, _ = kinds('https://a.example/x#t=1&u=' + quote('https://b.example/?z=2', safe=''))
        self.assertIn(('URL_DECODED', 'URL.fragment:percent-decode:1'), views)
        self.assertIn(('NESTED_URL', 'URL.fragment.u.query:percent-decode:2'), views)
        views, _ = kinds('https://a.example/x#https://c.example/?y=3')
        self.assertEqual(views, [('NESTED_URL', 'URL.fragment.query:percent-decode:2')])

    def test_url_first_seen_too_deep_is_decoded_where_allowed(self):
        target = 'https://t.example/?k=v'
        chain = target
        for host in ('l3', 'l2', 'l1'):
            chain = f'https://{host}.example/?a={quote(chain, safe="")}'
        views, gaps = kinds(f'https://a.example/?a={quote(chain, safe="")}&b={quote(target, safe="")}', max_depth=3)
        self.assertIn({'source': 'URL.query.a.query.a.query.a.query.a', 'gap': 'depth_exceeded'}, gaps)
        self.assertIn(('NESTED_URL', 'URL.query.b.query:percent-decode:2'), views)


if __name__ == '__main__':
    unittest.main()
