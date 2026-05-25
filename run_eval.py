"""run_eval.py — Harness principal de avaliação.

Roda cada prompt do golden dataset contra cada candidato OpenAI, avalia com
juiz LLM (gpt-4.1), e salva CSV em results/<run-tag>_<produto>.csv.

Uso:
  python run_eval.py
  python run_eval.py --product educiacao --run-tag run_v1
  python run_eval.py --product designmind --max-prompts 3   # debug rápido

Todos os candidatos e o juiz usam a mesma OPENAI_API_KEY (provider único).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from judge_prompt import build_judge_prompt, parse_judge_output, weighted_score

load_dotenv()

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_openai() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada. Configure em .env ou ambiente.")
        sys.exit(2)
    return OpenAI(api_key=api_key)


def run_candidate(
    client: OpenAI,
    model_id: str,
    prompt: str,
    max_tokens: int = 1200,
    retries: int = 2,
) -> dict:
    """Chama um candidato com retry exponencial. Retorna output, tokens, latência."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return {
                "output": r.choices[0].message.content,
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
                "latency_s": round(time.time() - t0, 3),
                "error": None,
            }
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} em {wait}s ({exc})")
            time.sleep(wait)
    return {
        "output": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_s": 0,
        "error": str(last_exc),
    }


def run_judge(client: OpenAI, judge_model: str, judge_prompt: str, rubric: dict) -> dict:
    """Chama o juiz e parseia a resposta JSON."""
    r = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=600,
        temperature=0.0,
    )
    raw = r.choices[0].message.content
    try:
        parsed = parse_judge_output(raw, rubric)
        parsed["_raw_judge_output"] = raw[:500]
        return parsed
    except ValueError as exc:
        return {
            "_error": str(exc),
            "_raw_judge_output": raw[:500],
        }


def resolve_rubric_path(product: str) -> Path:
    """Rubrica pode estar em data/ (Aula 06) ou em config/ (Aula 04/05)."""
    for parent in ("data", "config"):
        candidate = ROOT / parent / f"rubric_{product}.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Rubrica não encontrada: rubric_{product}.yaml (procurei em data/ e config/)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--product",
        default=os.environ.get("PRODUCT", "educiacao"),
        choices=["educiacao", "designmind"],
    )
    ap.add_argument("--run-tag", default="run_v1")
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="Limita N primeiros prompts (debug rápido)")
    args = ap.parse_args()

    print(f"\n[ aula06 · run_eval · OpenAI · produto={args.product} · tag={args.run_tag} ]\n")

    golden_path = ROOT / "data" / f"golden_{args.product}.csv"
    rubric_path = resolve_rubric_path(args.product)
    cand_path = ROOT / "config" / "candidates.yaml"

    golden = pd.read_csv(golden_path)
    if args.max_prompts:
        golden = golden.head(args.max_prompts)

    rubric = yaml.safe_load(open(rubric_path))
    cfg = yaml.safe_load(open(cand_path))
    candidates = cfg["candidates"]
    judge_model = cfg["judge"]["model_id"]

    judge_names = [c["name"] for c in candidates if c["model_id"] == judge_model]
    if judge_names:
        print(f"  AVISO: juiz '{judge_model}' também aparece como candidato "
              f"({judge_names}). Regra do harness: juiz ≠ candidato.")

    print(f"  Prompts: {len(golden)}  ·  Candidatos: {len(candidates)}  ·  Juiz: {judge_model}")
    print(f"  Total de chamadas estimado: {len(golden)*len(candidates)} candidatos + "
          f"{len(golden)*len(candidates)} juiz\n")

    client = make_openai()

    results = []
    student = os.environ.get("STUDENT_NAME", "anon")

    for cand in candidates:
        if cand.get("provider") != "openai":
            print(f"  [SKIP] {cand['name']}: provider '{cand.get('provider')}' não suportado")
            continue
        for _, row in golden.iterrows():
            print(f"  [{cand['name']:12s}] {row['id']} ({row['category']})...", end=" ", flush=True)

            cand_run = run_candidate(client, cand["model_id"], row["input"])

            if cand_run["error"]:
                print(f"FALHA · {cand_run['error']}")
                results.append({
                    "model": cand["name"], "id": row["id"], "category": row["category"],
                    "weighted": None, "error": cand_run["error"],
                })
                continue

            jp = build_judge_prompt(
                user_input=row["input"],
                candidate_output=cand_run["output"],
                expected_pattern=row["expected_pattern"],
                red_flags=row["red_flags"],
                rubric=rubric,
            )
            scores = run_judge(client, judge_model, jp, rubric)

            if "_error" in scores:
                print(f"JUIZ FALHOU · {scores['_error'][:80]}")
                weighted = None
            else:
                weighted = weighted_score(scores, rubric)
                print(f"weighted={weighted:.2f}  lat={cand_run['latency_s']:.2f}s  "
                      f"out={cand_run['output_tokens']}tok")

            row_out = {
                "model": cand["name"],
                "model_id": cand["model_id"],
                "id": row["id"],
                "category": row["category"],
                "weight_in_dataset": row["weight"],
                "weighted": weighted,
                "latency_s": cand_run["latency_s"],
                "input_tokens": cand_run["input_tokens"],
                "output_tokens": cand_run["output_tokens"],
                "error": cand_run["error"],
                "judge_error": scores.get("_error"),
            }
            for dim in rubric["dimensions"]:
                row_out[dim["name"]] = scores.get(dim["name"])
            row_out["red_flags_acionadas"] = json.dumps(
                scores.get("red_flags_acionadas", []), ensure_ascii=False
            )
            row_out["justificativa"] = scores.get("justificativa_curta", "")
            row_out["candidate_output"] = cand_run["output"][:2000]
            row_out["student"] = student
            row_out["run_tag"] = args.run_tag
            row_out["product"] = args.product
            results.append(row_out)

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / f"{args.run_tag}_{args.product}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Resultados salvos em {out_path.relative_to(ROOT)}")
    print(f"  Total de linhas: {len(df)}")
    if "weighted" in df.columns:
        print("\n  Média ponderada por modelo:")
        agg = df.groupby("model")["weighted"].mean().sort_values(ascending=False)
        for m, s in agg.items():
            print(f"    {m:15s}  {s:.3f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
