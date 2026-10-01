"""Small live smoke check using only the repository's fictional English corpus."""
import argparse
import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from pramana.api.runtime import build_pipeline
from pramana.config.settings import Settings
from pramana.generation.budget import request_budget
from pramana.schemas import Draft

logging.disable(logging.CRITICAL)
load_dotenv('.env', override=True)
report = {'timestamp': datetime.now(UTC).isoformat(), 'scope': 'fictional English corpus; live smoke only, not pilot acceptance', 'results': []}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--out', type=Path, default=Path('reports/pilot-hardening/live-smoke.json'),
    help='report path (defaults to reports/pilot-hardening/live-smoke.json)',
)
output = parser.parse_args().out
for provider in ('groq', 'google'):
    cfg = Settings(mode='pilot', api_key=secrets.token_urlsafe(32), corpus_dir=Path('examples/corpus'), providers=(provider,), languages=('en',), offline=False)
    pipeline, router, _ = build_pipeline(settings=cfg)
    try:
        for case in ('supported_answer', 'contradiction', 'missing_information'):
            started = perf_counter()
            row = {'provider': provider, 'case': case}
            try:
                with request_budget(24, 60):
                    if case == 'contradiction':
                        query = 'How many unused leave days may employees carry forward?'
                        retrieval = pipeline.retriever.retrieve(query, language='en', top_k=5)
                        claims = pipeline.decomposer.decompose(Draft(text='Employees may carry forward up to 90 unused leave days into the next year.', language='en'))
                        detection = pipeline.verifier.verify(claims, retrieval, language='en')
                        row.update(passed=detection.has_contradiction, verdicts=[v.verdict.value for v in detection.claim_verdicts])
                    else:
                        query = 'How many unused leave days may employees carry forward?' if case == 'supported_answer' else 'What is the reimbursement policy for employee journeys to Mars?'
                        result = pipeline.run(query, language='en')
                        passed = result.abstained if case == 'missing_information' else (not result.abstained and '15' in result.final_answer and result.detection.supported_ratio == 1 and bool(result.evidence_chunk_ids))
                        row.update(passed=passed, result=result.to_dict())
            except Exception as exc:
                row.update(passed=False, error_type=type(exc).__name__)
            row['elapsed_seconds'] = round(perf_counter() - started, 2)
            report['results'].append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({k: v for k, v in row.items() if k != 'result'}), flush=True)
    finally:
        router.close()
