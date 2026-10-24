"""Read the deliberately small OpenRouter credential format as data."""

import os
from pathlib import Path
import re


def load_api_key(path=Path('.env.openrouter'), variable='OPENROUTER_API_KEY'):
    value = os.environ.get(variable, '').strip()
    if not value:
        try:
            lines = [line.strip() for line in Path(path).read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
        except OSError:
            raise ValueError('OpenRouter credentials are missing; set OPENROUTER_API_KEY or .env.openrouter') from None
        if len(lines) != 1:
            raise ValueError('Credentials file must contain exactly one key assignment')
        match = re.fullmatch(re.escape(variable) + r'\s*=\s*(.*)', lines[0])
        if not match:
            raise ValueError('Malformed OpenRouter credentials assignment')
        value = match[1].strip()
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError('Malformed quoted OpenRouter credentials')
            value = value[1:-1]
    if not value or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('OpenRouter key is empty or malformed')
    return value
