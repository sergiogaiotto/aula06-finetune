"""analyze.py — Agrega resultados e gera a matriz de trade-offs.

Lê results/<run_tag>_<produto>.csv e produz:
  - results/<run_tag>_summary.md  (markdown com matriz de trade-offs)
  - imprime tabela no terminal

Uso:
  python analyze.py
  python analyze.py --product designmind --run-tag run_v1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from tabulate import tabulate

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"


def aggregate(df: pd.DataFrame, candidates: list[dict]) -> pd.DataFrame:
    rows = []
    cand_by_name = {c["name"]: c for c in candidates}

    for model, g in df.groupby("model"):
        valid = g.dropna(subset=["weighted"])
        if len(valid) == 0:
            continue

        n_total = len(g)
        n_failed = int(g["error"].notna().sum() + g["judge_error"].notna().sum())
        success_rate = (n_total - n_failed) / n_total if n_total else 0

        avg_w = valid["weighted"].mean()
        std_w = valid["weighted"].std()
        p50_lat = valid["latency_s"].quantile(0.5)
        p95_lat = valid["latency_s"].quantile(0.95)
        avg_in = valid["input_tokens"].mean()
        avg_out = valid["output_tokens"].mean()

        cand = cand_by_name.get(model, {})
        cost_in = cand.get("cost_per_1m_input_usd", 0)
        cost_out = cand.get("cost_per_1m_output_usd", 0)
        cost_per_1k_responses = (
            (avg_in * cost_in / 1_000_000) + (avg_out * cost_out / 1_000_000)
        ) * 1000

        red_flags_count = (
            valid["red_flags_acionadas"]
            .fillna("[]")
            .apply(lambda s: 0 if s in ("[]", "") else s.count(",") + 1)
            .sum()
        )

        rows.append({
            "model": model,
            "model_id": cand.get("model_id", ""),
            "perfil": cand.get("perfil", ""),
            "qualidade_avg": round(avg_w, 3),
            "qualidade_std": round(std_w, 3) if not pd.isna(std_w) else 0,
            "latencia_p50_s": round(p50_lat, 2),
            "latencia_p95_s": round(p95_lat, 2),
            "tokens_in_avg": int(avg_in),
            "tokens_out_avg": int(avg_out),
            "custo_usd_1k_resp": round(cost_per_1k_responses, 4),
            "red_flags_total": int(red_flags_count),
            "success_rate": round(success_rate, 2),
        })

    return pd.DataFrame(rows).sort_values("qualidade_avg", ascending=False)


def find_pareto(matrix: pd.DataFrame) -> set[str]:
    """Retorna set de modelos não-dominados (Pareto) em qualidade↑, custo↓, latência↓."""
    pareto = set()
    for i, row_i in matrix.iterrows():
        dominated = False
        for j, row_j in matrix.iterrows():
            if i == j:
                continue
            better_q = row_j["qualidade_avg"] >= row_i["qualidade_avg"]
            better_c = row_j["custo_usd_1k_resp"] <= row_i["custo_usd_1k_resp"]
            better_l = row_j["latencia_p95_s"] <= row_i["latencia_p95_s"]
            strictly = (
                row_j["qualidade_avg"] > row_i["qualidade_avg"]
                or row_j["custo_usd_1k_resp"] < row_i["custo_usd_1k_resp"]
                or row_j["latencia_p95_s"] < row_i["latencia_p95_s"]
            )
            if better_q and better_c and better_l and strictly:
                dominated = True
                break
        if not dominated:
            pareto.add(row_i["model"])
    return pareto


def write_summary(matrix: pd.DataFrame, pareto: set, out_path: Path,
                  product: str, run_tag: str, df_raw: pd.DataFrame) -> None:
    lines = [
        f"# Resumo da execução · {run_tag} · {product}",
        "",
        f"- Linhas válidas: {len(df_raw.dropna(subset=['weighted']))}",
        f"- Linhas com falha: {int(df_raw['error'].notna().sum() + df_raw['judge_error'].notna().sum())}",
        "",
        "## Matriz de trade-offs",
        "",
        tabulate(matrix, headers="keys", tablefmt="github", showindex=False),
        "",
        "## Conjunto Pareto-ótimo",
        "",
        f"Modelos não-dominados (qualidade↑, custo↓, latência↓): **{', '.join(sorted(pareto))}**",
        "",
        "## Quebra por categoria",
        "",
    ]
    cat_pivot = (
        df_raw.dropna(subset=["weighted"])
        .pivot_table(index="category", columns="model", values="weighted", aggfunc="mean")
        .round(2)
    )
    lines.append(tabulate(cat_pivot, headers="keys", tablefmt="github"))
    lines += [
        "",
        "## Próximos passos (Fase 6 → 7)",
        "",
        "1. Revisar amostra de outputs absurdos vs. score alto (juiz pode estar leniente).",
        "2. Decidir o campeão considerando restrição operacional do produto.",
        "3. Documentar trade-off doloroso aceito.",
        "4. Preencher `docs/model_card_v1.md`.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--product",
        default=os.environ.get("PRODUCT", "educiacao"),
        choices=["educiacao", "designmind"],
    )
    ap.add_argument("--run-tag", default="run_v1")
    args = ap.parse_args()

    csv_path = RESULTS_DIR / f"{args.run_tag}_{args.product}.csv"
    if not csv_path.exists():
        print(f"  Arquivo não encontrado: {csv_path}")
        print("  Rode antes: python run_eval.py")
        return 1

    df = pd.read_csv(csv_path)
    cfg = yaml.safe_load(open(ROOT / "config" / "candidates.yaml"))

    matrix = aggregate(df, cfg["candidates"])
    pareto = find_pareto(matrix)

    print(f"\n[ aula04 · analyze · {args.product} · {args.run_tag} ]\n")
    print(tabulate(matrix, headers="keys", tablefmt="grid", showindex=False))
    print(f"\n  Pareto-ótimo: {sorted(pareto)}\n")

    out = RESULTS_DIR / f"{args.run_tag}_{args.product}_summary.md"
    write_summary(matrix, pareto, out, args.product, args.run_tag, df)
    print(f"  Summary salvo em {out.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
