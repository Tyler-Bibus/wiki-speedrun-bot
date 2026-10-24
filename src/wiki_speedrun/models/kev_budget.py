"""Weight-free CPU budgeting using the pinned upstream Kev encoder itself.

Only pure encoding definitions are compiled from the installed, revision-checked
source. This avoids importing kev.model (which imports torch and model loaders).
The preserved tokenizer.json is loaded by the existing optional tokenizers wheel.
"""
import ast
import importlib.metadata
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

from .trainable_kev import KEV_REVISION, _installed_kev_revision
from ..training.checkpoints import sha256_file


def encoder_namespace():
    if _installed_kev_revision() != KEV_REVISION:
        raise RuntimeError('Matched budgeting requires the pinned Kev source revision')
    root = Path(importlib.util.find_spec('kev').origin).parent
    path = root / 'model.py'
    names = {'ContextOverflow', 'user_tokens', 'encode'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise RuntimeError('Pinned Kev encoding definitions are missing')
    namespace = {'re': re, '_SPECIAL_RE': re.compile(r'<\|([A-Za-z0-9_]+)\|>'),
                 'MAX_STATE': 384, 'MAX_BRANCH': 1024, 'OPT_NONE': -1, 'OPT_DECIDE': -2,
                 'SPECIAL': ['<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, root


class KevContextBudgeter:
    def __init__(self, checkpoint, config):
        from tokenizers import Tokenizer
        self.namespace, source = encoder_namespace()
        path = Path(checkpoint)
        if not path.is_dir() or not (path / 'tokenizer.json').is_file():
            raise ValueError('Matched budgeting needs a preserved local tokenizer directory')
        tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))

        class TokenizerAdapter:
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

            def convert_tokens_to_ids(self, text):
                value = tokenizer.token_to_id(text)
                if value is None:
                    raise ValueError('Tokenizer lacks Kev control tokens')
                return value

        self.tokenizer = TokenizerAdapter()
        self.max_state_tokens = config.max_state_tokens
        self.max_row_tokens = config.max_row_tokens
        self.metadata = {'kev_revision': KEV_REVISION, 'tokenizer_directory': str(path),
                         'tokenizer_sha256': sha256_file(path / 'tokenizer.json'),
                         'encoding_sha256': sha256_file(source / 'model.py'),
                         'api_sha256': sha256_file(source / 'api.py'),
                         'tokenizers_version': importlib.metadata.version('tokenizers')}

    def state_token_count(self, state):
        from kev.api import render
        return len(self.namespace['user_tokens'](self.tokenizer, render(state))) + 1

    def token_counts(self, request):
        from kev.api import SystemOneRequest, to_record
        record, _ = to_record(SystemOneRequest.model_validate(request))
        result = self.namespace['encode'](self.tokenizer, record, max_state=self.max_state_tokens,
                                         max_branch=self.max_row_tokens, strict=True)
        return {'state': result['state_tokens'], 'row': len(result['ids'])}

    def fits(self, request):
        try:
            self.token_counts(request)
            return True
        except self.namespace['ContextOverflow']:
            return False
