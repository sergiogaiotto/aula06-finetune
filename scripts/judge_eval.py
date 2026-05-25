"""scripts/judge_eval.py — Pontua respostas fine-tuned usando rubrica via OpenAI.

Migrado para OpenAI API (gpt-4.1 como juiz).
Lê CSV com colunas id, input, output_finetuned. Junta com golden para
recuperar expected_pattern e red_flags. Aplica juiz LLM com rubrica.
Opcionalmente compara com baseline da Aula 04.

Uso:
  python scripts/judge_eval.py \
    --eval-csv results/eval_finetuned.csv \
    --golden-csv data/golden_educiacao.csv \
    --rubric-yaml data/rubric_educiacao.yaml \
    --baseline-csv data/eval_baseline_educiacao.csv \
    --judge-model gpt-4.1 \
    --out results/eval_finetuned_scored.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm


def build_judge_prompt(
    user_input: str,
    candidate_output: str,
    expected_pattern: str,
    red_flags: str,
    rubric: dict,
) -> str:
    """Prompt do juiz com âncoras da escala e mitigações dos 5 vieses."""
    dims_block = []
    for dim in rubric["dimensions"]:
        scale = "\n".join(f"      {k}: {v}" for k, v in dim["scale"].items())
        dims_block.append(
            f"  - id: {dim['name']}\n"
            f"    label: {dim['label']}\n"
            f"    weight: {dim['weight']}\n"
            f"    description: {dim['description']}\n"
            f"    scale:\n{scale}"
        )
    dims_yaml = "\n".join(dims_block)

    schema = {dim["name"]: "<int 1-5>" for dim in rubric["dimensions"]}
    schema["red_flags_acionadas"] = ["<string>"]
    schema["justificativa_curta"] = "<string até 80 palavras>"

    return textwrap.dedent(f"""
    Você é um juiz especialista. Avalie a RESPOSTA de um modelo fine-tuned
    em relação a um INPUT do usuário, usando a RUBRICA com dimensões ponderadas.

    REGRAS CRÍTICAS:
    - Avalie estritamente pelas âncoras da escala 1-5 de cada dimensão.
    - NÃO premie respostas mais longas. Tamanho não é qualidade.
    - NÃO premie formatação rica se o conteúdo é raso.
    - Se a resposta acionar uma RED FLAG, a dimensão correspondente cai para 1 ou 2.
    - Saída exclusivamente JSON. Sem prefixo, sem sufixo, sem markdown wrapper.

    INPUT DO USUÁRIO
    ---
    {user_input}
    ---

    RESPOSTA DO MODELO FINE-TUNED
    ---
    {candidate_output}
    ---

    PADRÃO DE RESPOSTA ESPERADA
    ---
    {expected_pattern}
    ---

    RED FLAGS
    ---
    {red_flags}
    ---

    RUBRICA
    ---
    {dims_yaml}
    ---

    SCHEMA DA SAÍDA (JSON estrito)
    ---
    {json.dumps(schema, indent=2, ensure_ascii=False)}
    ---

    Retorne agora apenas o JSON.
    """).strip()


def parse_judge_output(raw: str, rubric: dict) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    parsed = json.loads(text)
    expected = {dim["name"] for dim in rubric["dimensions"]}
    missing = expected - set(parsed.keys())
    if missing:
        raise ValueError(f"JSON do juiz faltando dimensões: {missing}")
    return parsed


def weighted_score(scores: dict, rubric: dict) -> float:
    total = 0.0
    for dim in rubric["dimensions"]:
        nota = scores.get(dim["name"])
        if nota is None:
            continue
        total += float(nota) * float(dim["weight"])
    return round(total, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True,
                    help="CSV com id, input, output_finetuned")
    ap.add_argument("--golden-csv", required=True,
                    help="Golden dataset com expected_pattern e red_flags")
    ap.add_argument("--rubric-yaml", required=True)
    ap.add_argument("--baseline-csv", default=None,
                    help="Opcional: CSV com scores da Aula 04 para comparar")
    ap.add_argument("--judge-model", default="gpt-4.1",
                    help="Modelo juiz na OpenAI (default: gpt-4.1)")
    ap.add_argument("--out", default="eval_finetuned_scored.csv")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada.")
        return 1

    eval_df = pd.read_csv(args.eval_csv)
    golden = pd.read_csv(args.golden_csv).set_index("id")
    rubric = yaml.safe_load(open(args.rubric_yaml))

    client = OpenAI(api_key=api_key)

    print(f"\n[ aula06 · judge_eval · OpenAI ]")
    print(f"  Eval:    {args.eval_csv} ({len(eval_df)} respostas)")
    print(f"  Juiz:    {args.judge_model}")
    print(f"  Custo estimado: ~USD {len(eval_df) * 0.008:.2f}\n")

    rows = []
    for _, r in tqdm(eval_df.iterrows(), total=len(eval_df), desc="  julgando"):
        seed_id = r["id"]
        if seed_id not in golden.index:
            tqdm.write(f"    [skip] {seed_id} sem entrada no golden")
            continue
        seed = golden.loc[seed_id]
        prompt = build_judge_prompt(
            user_input=r["input"],
            candidate_output=r["output_finetuned"],
            expected_pattern=seed["expected_pattern"],
            red_flags=seed["red_flags"],
            rubric=rubric,
        )
        try:
            resp = client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.0,
            )
            scores = parse_judge_output(resp.choices[0].message.content, rubric)
            weighted = weighted_score(scores, rubric)
            row_out = {
                "id": seed_id,
                "weighted_finetuned": weighted,
            }
            for dim in rubric["dimensions"]:
                row_out[f"{dim['name']}_finetuned"] = scores.get(dim["name"])
            row_out["red_flags_acionadas"] = json.dumps(
                scores.get("red_flags_acionadas", []), ensure_ascii=False
            )
            row_out["justificativa"] = scores.get("justificativa_curta", "")
            rows.append(row_out)
        except Exception as exc:
            tqdm.write(f"    [fail] {seed_id}: {exc}")
            rows.append({"id": seed_id, "weighted_finetuned": None, "error": str(exc)})

    df_scored = pd.DataFrame(rows)

    if args.baseline_csv:
        try:
            baseline = pd.read_csv(args.baseline_csv)
            if "model" in baseline.columns:
                baseline_agg = baseline.groupby("id")["weighted"].mean().rename("weighted_baseline")
            else:
                baseline_agg = baseline.set_index("id")["weighted"].rename("weighted_baseline")
            df_scored = df_scored.merge(baseline_agg, left_on="id", right_index=True, how="left")
            df_scored["delta"] = df_scored["weighted_finetuned"] - df_scored["weighted_baseline"]
        except Exception as exc:
            print(f"  [warn] não foi possível carregar baseline: {exc}")

    df_scored.to_csv(args.out, index=False)
    print(f"\n  Output: {args.out}")
    print("\n  Média ponderada · fine-tuned:", round(df_scored["weighted_finetuned"].mean(), 3))
    if "weighted_baseline" in df_scored.columns:
        print("  Média ponderada · baseline:  ", round(df_scored["weighted_baseline"].mean(), 3))
        print("  Delta médio:                 ",
              round((df_scored["weighted_finetuned"] - df_scored["weighted_baseline"]).mean(), 3))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
