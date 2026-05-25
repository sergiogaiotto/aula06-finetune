"""scripts/compare_baseline.py — Tabela comparativa baseline vs fine-tuned.

Gera o relatório final que o aluno cola no Model Card v3:
  * Score ponderado total: baseline → fine-tuned → delta
  * Score por dimensão da rubrica
  * Dimensões que melhoraram, que regrediram (catastrophic forgetting)
  * Casos onde fine-tuned superou e onde piorou

Uso:
  python scripts/compare_baseline.py \\
    --scored eval_finetuned_scored.csv \\
    --rubric rubric_educiacao.yaml \\
    --out comparison_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from tabulate import tabulate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True,
                    help="CSV de saída do judge_eval.py")
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--out", default="comparison_report.md")
    args = ap.parse_args()

    df = pd.read_csv(args.scored)
    rubric = yaml.safe_load(open(args.rubric))
    dims = [d["name"] for d in rubric["dimensions"]]

    if "weighted_baseline" not in df.columns:
        print("ERRO: CSV não contém weighted_baseline. Rode judge_eval.py com --baseline-csv.")
        return 1

    valid = df.dropna(subset=["weighted_finetuned", "weighted_baseline"]).copy()

    avg_ft = valid["weighted_finetuned"].mean()
    avg_bl = valid["weighted_baseline"].mean()
    delta = avg_ft - avg_bl

    # Comparação por dimensão
    dim_table = []
    for dim_name in dims:
        col_ft = f"{dim_name}_finetuned"
        if col_ft not in valid.columns:
            continue
        ft_avg = valid[col_ft].mean()
        dim_table.append({
            "dimensão": dim_name,
            "fine-tuned": round(ft_avg, 2),
            "peso": next(d["weight"] for d in rubric["dimensions"] if d["name"] == dim_name),
        })

    dim_df = pd.DataFrame(dim_table)

    # Por query
    query_table = valid[["id", "weighted_baseline", "weighted_finetuned"]].copy()
    query_table["delta"] = (query_table["weighted_finetuned"]
                             - query_table["weighted_baseline"]).round(3)
    query_table = query_table.sort_values("delta", ascending=False)

    melhores = query_table.head(3)
    piores = query_table.tail(3)

    # Catastrophic forgetting: queries onde fine-tuned piorou
    forgetting = query_table[query_table["delta"] < -0.3]

    # Relatório markdown
    lines = [
        "# Comparativo Baseline vs Fine-Tuned",
        "",
        "## Score ponderado total",
        "",
        f"- Baseline (Aula 04):  **{avg_bl:.3f}**",
        f"- Fine-tuned (Aula 06): **{avg_ft:.3f}**",
        f"- Delta: **{delta:+.3f}** ({'✓ melhora' if delta > 0 else '✗ regressão'})",
        "",
        "## Score por dimensão (fine-tuned)",
        "",
        tabulate(dim_df, headers="keys", tablefmt="github", showindex=False),
        "",
        "## Top 3 queries com maior ganho",
        "",
        tabulate(melhores, headers="keys", tablefmt="github", showindex=False),
        "",
        "## Bottom 3 queries (maior perda ou menor ganho)",
        "",
        tabulate(piores, headers="keys", tablefmt="github", showindex=False),
        "",
    ]

    if len(forgetting) > 0:
        lines.extend([
            "## ⚠ Catastrophic forgetting detectado",
            "",
            f"**{len(forgetting)} queries** tiveram regressão > 0.3 pontos:",
            "",
            tabulate(forgetting, headers="keys", tablefmt="github", showindex=False),
            "",
            "**Ação requerida**: declarar no Model Card v3 e considerar:",
            "- Reduzir num_train_epochs",
            "- Misturar 10-20% de dados gerais no próximo dataset",
            "- Reduzir learning rate",
            "",
        ])
    else:
        lines.extend([
            "## ✓ Sem catastrophic forgetting significativo",
            "",
            "Nenhuma query regrediu mais de 0.3 pontos.",
            "",
        ])

    lines.extend([
        "## Decisão para o Model Card v3",
        "",
        f"- Promover o adapter? **{'SIM' if delta > 0.2 else 'NÃO' if delta < 0 else 'REVISAR'}**",
        f"- Critério: delta médio = {delta:+.3f} (limite mínimo razoável: +0.2)",
        "",
    ])

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Relatório salvo em {args.out}")
    print(f"  Delta: {delta:+.3f}")
    print(f"  Catastrophic forgetting: {len(forgetting)} queries\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
