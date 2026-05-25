"""judge_prompt.py — Construção do prompt do juiz LLM com mitigação dos 5 vieses.

Vieses que o juiz LLM apresenta, segundo Zheng et al. 2023 (MT-Bench):
  1. Position bias    — primeira resposta tende a ganhar
  2. Length bias      — respostas longas parecem melhores
  3. Self-preference  — modelo prefere outputs do mesmo modelo
  4. Authority bias   — tom autoritativo confunde com correção
  5. Verbosity-as-quality — formatação rica parece qualidade

Mitigações implementadas:
  * Pointwise scoring com âncoras de escala explícitas (mitiga 2, 5)
  * Instrução anti-comprimento ("não premiar por tamanho")
  * Red flags como hard fails (corta 4)
  * Juiz ≠ candidato (regra do harness, não do prompt)
  * Pairwise opcional com swap de ordem (mitiga 1, modo experimental)
"""
from __future__ import annotations

import json
import textwrap


def build_judge_prompt(
    user_input: str,
    candidate_output: str,
    expected_pattern: str,
    red_flags: str,
    rubric: dict,
) -> str:
    """Monta o prompt pointwise do juiz com âncoras de escala."""

    dims_block = []
    for dim in rubric["dimensions"]:
        scale = "\n".join(
            f"      {k}: {v}" for k, v in dim["scale"].items()
        )
        dims_block.append(
            f"  - id: {dim['name']}\n"
            f"    label: {dim['label']}\n"
            f"    weight: {dim['weight']}\n"
            f"    description: {dim['description']}\n"
            f"    scale:\n{scale}"
        )
    dims_yaml = "\n".join(dims_block)

    schema = {dim["name"]: "<int 1-5>" for dim in rubric["dimensions"]}
    schema["red_flags_acionadas"] = ["<string>", "<string>"]
    schema["justificativa_curta"] = "<string até 80 palavras>"

    return textwrap.dedent(f"""
    Você é um juiz especialista. Sua tarefa é avaliar a RESPOSTA de um modelo
    candidato em relação a uma INPUT do usuário, usando uma RUBRICA com
    dimensões ponderadas.

    REGRAS CRÍTICAS:
    - Avalie estritamente pelas âncoras da escala 1-5 de cada dimensão.
    - NÃO premie respostas mais longas. Tamanho não é qualidade.
    - NÃO premie formatação rica (bullets, bold) se o conteúdo é raso.
    - Se a resposta acionar uma RED FLAG, a dimensão correspondente cai para 1
      ou 2 obrigatoriamente, mesmo que o resto pareça bem feito.
    - Sua saída DEVE ser exclusivamente um JSON válido. Sem prefixo, sem sufixo,
      sem markdown wrapper. Comece com {{ e termine com }}.

    INPUT DO USUÁRIO
    ---
    {user_input}
    ---

    RESPOSTA DO CANDIDATO
    ---
    {candidate_output}
    ---

    PADRÃO DE RESPOSTA ESPERADA (referência, não substitui rubrica)
    ---
    {expected_pattern}
    ---

    RED FLAGS QUE REPROVAM AUTOMATICAMENTE (forçam nota 1 ou 2 nas dimensões afetadas)
    ---
    {red_flags}
    ---

    RUBRICA (avalie cada dimensão na escala 1-5)
    ---
    {dims_yaml}
    ---

    SCHEMA DA SUA SAÍDA (JSON estrito)
    ---
    {json.dumps(schema, indent=2, ensure_ascii=False)}
    ---

    Retorne agora apenas o JSON.
    """).strip()


def parse_judge_output(raw: str, rubric: dict) -> dict:
    """Parse robusto do JSON do juiz, com tolerância a code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Juiz não retornou JSON válido: {exc}\nRaw: {raw[:300]}")
    expected_keys = {dim["name"] for dim in rubric["dimensions"]}
    missing = expected_keys - set(parsed.keys())
    if missing:
        raise ValueError(f"JSON do juiz não tem dimensões: {missing}")
    return parsed


def weighted_score(scores: dict, rubric: dict) -> float:
    """Soma ponderada das notas pelas pesos da rubrica."""
    total = 0.0
    for dim in rubric["dimensions"]:
        nota = scores.get(dim["name"])
        if nota is None:
            continue
        total += float(nota) * float(dim["weight"])
    return round(total, 3)
