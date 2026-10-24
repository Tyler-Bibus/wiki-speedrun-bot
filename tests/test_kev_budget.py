"""Acceptance gate against the real pinned local encoder, without model weights."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from wiki_speedrun.choices import _fit_state, _groups, make_request
from wiki_speedrun.remote_config import RemoteConfig


@unittest.skipUnless(Path('checkpoints/20261022T025733.517853Z-sft/update-001700/tokenizer.json').is_file()
                     and importlib.util.find_spec('tokenizers') and importlib.util.find_spec('kev'),
                     'preserved tokenizer and optional Kev installation required')
class RealEncoderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wiki_speedrun.models.kev_budget import KevContextBudgeter
        from wiki_speedrun.models.trainable_kev import TrainableKev
        from transformers import AutoTokenizer
        from kev.model import encode
        cls.config = RemoteConfig()
        cls.remote = KevContextBudgeter(cls.config.tokenizer, cls.config)
        tok = AutoTokenizer.from_pretrained(cls.config.tokenizer, local_files_only=True)
        cls.local = TrainableKev(tok, SimpleNamespace(encode=encode), {}, 1024, 4096)

    def test_fixture_long_states_and_control_tokens(self):
        for excerpt in ('Earth', '<|fim_prefix|> <|box_end|> Ω 日', 'long state ' * 3000):
            state = {'current_title': 'Earth', 'target_title': 'Moon', 'current_excerpt': excerpt,
                     'history': ['Earth'] * 100}
            left, ls = _fit_state(self.local, state, self.config)
            right, rs = _fit_state(self.remote, state, self.config)
            self.assertEqual((left, ls), (right, rs))
            req = make_request(left, ['link_0', 'link_1'], ['A', 'B'])
            self.assertEqual(self.local.state_token_count(left), self.remote.state_token_count(left))
            self.assertEqual(self.local.token_counts(req), self.remote.token_counts(req))

    def test_boundary_and_large_group_membership(self):
        ids = [f'link_{i}' for i in range(301)]
        articles = {i: ('long_article_title_' * (index % 30 + 1)) for index, i in enumerate(ids)}
        state = {'current_title': 'A', 'target_title': 'B', 'history': [], 'current_excerpt': 'x' * 1500}
        state, _ = _fit_state(self.remote, state, self.config)
        self.assertEqual(_groups(self.local, state, ids, articles, self.config),
                         _groups(self.remote, state, ids, articles, self.config))
        request = make_request(state, ids[:2], [articles[k] for k in ids[:2]])
        tokens = self.local.token_counts(request)
        for delta in (-1, 0, 1):
            self.local.max_row_tokens = self.remote.max_row_tokens = tokens['row'] + delta
            self.assertEqual(self.local.fits(request), self.remote.fits(request))
        self.local.max_row_tokens = self.remote.max_row_tokens = 4096
        for delta in (-1, 0, 1):
            self.local.max_state_tokens = self.remote.max_state_tokens = tokens['state'] + delta
            self.assertEqual(self.local.fits(request), self.remote.fits(request))
        self.local.max_state_tokens = self.remote.max_state_tokens = 1024

    def test_remote_budget_process_does_not_import_torch(self):
        code = ('import sys; from wiki_speedrun.models.kev_budget import KevContextBudgeter; '
                'from wiki_speedrun.remote_config import RemoteConfig; c=RemoteConfig(); '
                'b=KevContextBudgeter(c.tokenizer,c); b.state_token_count({"x":"text"}); '
                'assert "torch" not in sys.modules; assert "transformers" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
