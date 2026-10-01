"""Run the localhost faculty demo with live APIs or explicitly selected offline fixtures."""
from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaunchPlan:
    port: int
    providers: tuple[str, ...]
    semantic: bool


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--provider', choices=('auto', 'google', 'groq'), default='auto')
    parser.add_argument('--offline', action='store_true', help='Use local fixtures without a model key.')
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument('--semantic', dest='semantic', action='store_const', const=True, default=None,
                          help='Use Google API embeddings; default auto-enables them when a Google key exists.')
    semantic.add_argument('--no-semantic', dest='semantic', action='store_const', const=False,
                          help='Use sparse search only, without embedding API calls.')
    parser.add_argument('--check', action='store_true', help='Check local configuration without starting or calling providers.')
    parser.add_argument('--container', action='store_true', help='Container-only bind; Compose must publish on loopback.')
    parser.add_argument('--live-google', action='store_true',
                        help='Backward-compatible alias for --provider google.')
    return parser


def resolve_plan(args: argparse.Namespace, parser: argparse.ArgumentParser) -> LaunchPlan:
    try:
        port = args.port if args.port is not None else int(os.getenv('PRAMANA_PORT', '8765'))
    except ValueError:
        parser.error('PRAMANA_PORT must be a number between 1 and 65535')
    if not 1 <= port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.offline and (args.live_google or args.provider != 'auto' or args.semantic is True):
        parser.error('--offline cannot be combined with a live provider or --semantic')
    requested = 'google' if args.live_google else args.provider
    configured = {'google': bool(os.getenv('GOOGLE_API_KEY', '').strip()),
                  'groq': bool(os.getenv('GROQ_API_KEY', '').strip())}
    if args.offline:
        providers = ('stub',)
    elif requested == 'auto':
        providers = tuple(name for name in ('google', 'groq') if configured[name])
        if not providers:
            parser.error('Set GOOGLE_API_KEY or GROQ_API_KEY in .env; use --offline explicitly for fixture tests.')
    else:
        if not configured[requested]:
            parser.error(f'{requested.upper()}_API_KEY is missing from .env')
        providers = (requested,)
    setting = os.getenv('PRAMANA_SEMANTIC', 'auto').strip().lower()
    if setting not in {'auto', 'true', 'false', '1', '0'}:
        parser.error('PRAMANA_SEMANTIC must be auto, true or false')
    semantic = args.semantic
    if semantic is None:
        semantic = not args.offline and (configured['google'] if setting == 'auto' else setting in {'true', '1'})
    if semantic and (args.offline or not configured['google']):
        parser.error('Semantic search requires online mode and GOOGLE_API_KEY; use --no-semantic for sparse search')
    return LaunchPlan(port, providers, bool(semantic))


def available_port(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            connection.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False


def bind_host(container: bool) -> str:
    if container and not Path('/.dockerenv').is_file():
        raise ValueError('--container is only allowed inside Docker; local demos stay on loopback')
    return '0.0.0.0' if container else '127.0.0.1'


def configure_environment(plan: LaunchPlan) -> Path:
    live = plan.providers != ('stub',)
    corpus = Path(__file__).resolve().parents[1] / 'examples' / 'corpus'
    os.environ.update({
        'PRAMANA_MODE': 'demo', 'PRAMANA_OFFLINE': 'false' if live else 'true',
        'PRAMANA_TRUSTED_HOSTS': '127.0.0.1,localhost,::1',
        'PRAMANA_PROVIDERS': ','.join(plan.providers), 'PRAMANA_API_KEY': '',
        'PRAMANA_CORPUS_DIR': str(corpus), 'PRAMANA_LANGUAGES': 'en,hi,ta',
        'PRAMANA_DENSE_MODEL': '', 'PRAMANA_CONFIDENCE_MODEL': '',
        'PRAMANA_API_EMBEDDING_MODEL': 'gemini-embedding-001' if plan.semantic else '',
        'PRAMANA_PROVIDER_CONFIG': '', 'PRAMANA_CACHE_ENABLED': 'false',
        'PRAMANA_EXPOSE_CORPUS': 'true',
    })
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    return corpus


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        host = bind_host(args.container)
    except ValueError as exc:
        parser.error(str(exc))
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    plan = resolve_plan(args, parser)
    configure_environment(plan)
    from pramana.config.settings import Settings
    from pramana.ingestion.corpus import load_corpus

    try:
        cfg = Settings.from_env()
        corpus = load_corpus(cfg.corpus_dir, cfg.languages)
    except (ValueError, OSError):
        parser.error('Configuration/corpus is invalid. Check .env and examples/corpus; no provider was called.')
    if not available_port(plan.port):
        parser.error(f'Local port {plan.port} is busy or unavailable. Stop the existing server or use --port 8770.')
    live = plan.providers != ('stub',)
    print(f'PRAMANA: http://127.0.0.1:{plan.port}/', flush=True)
    print(f'Mode: {"offline fixture (not live answers)" if not live else "live " + " -> ".join(plan.providers)}', flush=True)
    print(f'Retrieval: {"Google API semantic + sparse" if plan.semantic else "sparse (no embeddings)"}; languages: en, hi, ta', flush=True)
    print(f'Corpus loaded: {sum(len(chunks) for chunks in corpus.values())} fictional sample chunks. No local models.', flush=True)
    if args.check:
        print('Local checks passed. Provider access/quota and live answer quality were not tested.', flush=True)
        return
    if plan.semantic:
        print('Questions and document text are sent to Google for embeddings; live answers use the selected APIs.', flush=True)
    elif live:
        print('Questions and retrieved document text are sent to the selected APIs.', flush=True)
    print('Open the URL above. Keep this terminal open; Ctrl+C stops PRAMANA.', flush=True)
    import uvicorn

    uvicorn.run('pramana.api.service:app', host=host,
                port=plan.port, workers=1, proxy_headers=False, timeout_keep_alive=5)


if __name__ == '__main__':
    main()
