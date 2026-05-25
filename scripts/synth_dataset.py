"""scripts/synth_dataset.py — Síntese standalone fora do Colab.

Usa OpenAI API (gpt-4.1-mini como teacher por default).
Permite gerar o dataset sintético em qualquer máquina com Python + OPENAI_API_KEY.

Uso:
  python scripts/synth_dataset.py \
    --seeds data/golden_educiacao.csv \
    --produto "EducIAção (tutor pedagógico)" \
    --teacher gpt-4.1-mini \
    --n-per-seed 10 \
    --out data/dataset_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
PROMPT_PATH = ROOT / "prompts" / "synthesis_template.txt"


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def synthesize_one(
    client: OpenAI,
    template: str,
    row: pd.Series,
    produto: str,
    teacher_model: str,
    tam_min: int = 100,
    tam_max: int = 300,
) -> dict | None:
    """Gera 1 exemplo sintético a partir de uma seed via OpenAI."""
    prompt = template.format(
        produto=produto,
        tam_min=tam_min,
        tam_max=tam_max,
        original=row["input"],
        expected=row["expected_pattern"],
        red_flags=row["red_flags"],
    )
    try:
        r = client.chat.completions.create(
            model=teacher_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.8,
        )
        ex = json.loads(r.choices[0].message.content)
        if not ex.get("input") or not ex.get("output"):
            return None
        ex["source_seed"] = row["id"]
        ex["category"] = row.get("category", "")
        return ex
    except Exception as exc:
        print(f"    [warn] falha: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="CSV de seeds (golden dataset Aula 04)")
    ap.add_argument("--produto", required=True, help="Nome humano do produto")
    ap.add_argument("--teacher", default="gpt-4.1-mini",
                    help="Modelo teacher na OpenAI (default: gpt-4.1-mini)")
    ap.add_argument("--n-per-seed", type=int, default=10,
                    help="Variações geradas por seed (default: 10 → 50 seeds × 10 = 500 exemplos)")
    ap.add_argument("--tam-min", type=int, default=100)
    ap.add_argument("--tam-max", type=int, default=300)
    ap.add_argument("--out", default="dataset_raw.jsonl")
    ap.add_argument("--sleep-ms", type=int, default=50,
                    help="Pausa entre chamadas (rate limit)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada. Configure em .env ou ambiente.")
        return 1

    client = OpenAI(api_key=api_key)

    seeds = pd.read_csv(args.seeds)
    template = load_prompt_template()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[ aula06 · synth_dataset · OpenAI ]")
    print(f"  Seeds:    {args.seeds} ({len(seeds)} prompts)")
    print(f"  Teacher:  {args.teacher}")
    print(f"  Produto:  {args.produto}")
    print(f"  Total alvo: {len(seeds) * args.n_per_seed} exemplos")
    print(f"  Output:   {out_path}\n")

    cost_est = len(seeds) * args.n_per_seed * 0.0014
    print(f"  Custo estimado: ~USD {cost_est:.2f}\n")

    dataset = []
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in seeds.iterrows():
            print(f"  expandindo {row['id']}...")
            for _ in tqdm(range(args.n_per_seed), desc=f"    {row['id']}", leave=False):
                ex = synthesize_one(
                    client, template, row, args.produto, args.teacher,
                    args.tam_min, args.tam_max,
                )
                if ex:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    f.flush()
                    dataset.append(ex)
                time.sleep(args.sleep_ms / 1000)

    print(f"\n  Gerados:  {len(dataset)} exemplos")
    print(f"  Salvos em: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
