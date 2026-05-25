"""tests/test_smoke_colab.py — Smoke test do ambiente standalone (fora do Colab).

Valida que OpenAI key está configurada e funcional, e que as dependências
necessárias para os scripts auxiliares estão disponíveis.
"""
import os
import sys


def test_imports():
    """Confere que as bibliotecas-chave estão instaladas."""
    try:
        import openai      # noqa: F401
        import pandas      # noqa: F401
        import yaml        # noqa: F401
        import tqdm        # noqa: F401
        from dotenv import load_dotenv  # noqa: F401
        print("  [imports] OK")
        return True
    except ImportError as exc:
        print(f"  [imports] FALHA: {exc}")
        print("            rode: pip install -r requirements.txt")
        return False


def test_env_vars():
    """Verifica chave OpenAI."""
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [env] AUSENTE: OPENAI_API_KEY")
        return False
    print("  [env] OK (chave presente)")
    return True


def test_prompts_file():
    """Confere que o template de síntese está presente."""
    from pathlib import Path
    p = Path(__file__).parent.parent / "prompts" / "synthesis_template.txt"
    if not p.exists():
        print(f"  [prompts] FALHA: {p} ausente")
        return False
    txt = p.read_text(encoding="utf-8")
    required = ["{produto}", "{original}", "{expected}", "{red_flags}",
                "{tam_min}", "{tam_max}"]
    missing = [ph for ph in required if ph not in txt]
    if missing:
        print(f"  [prompts] FALHA: placeholders ausentes: {missing}")
        return False
    print("  [prompts] OK")
    return True


def test_openai_api():
    """Faz chamada teste à OpenAI com modelo barato (nano)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        r = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "diga apenas: pong"}],
            max_tokens=10,
        )
        print(f"  [openai] OK · {r.choices[0].message.content[:30]!r}")
        return True
    except Exception as exc:
        print(f"  [openai] FALHA: {exc}")
        return False


def test_data_files():
    """Verifica que data files necessários para os 2 produtos existem."""
    from pathlib import Path
    base = Path(__file__).parent.parent / "data"
    required = [
        "golden_educiacao.csv", "golden_designmind.csv",
        "rubric_educiacao.yaml", "rubric_designmind.yaml",
        "eval_baseline_educiacao.csv", "eval_baseline_designmind.csv",
    ]
    missing = [f for f in required if not (base / f).exists()]
    if missing:
        print(f"  [data] FALHA: arquivos ausentes: {missing}")
        return False
    print(f"  [data] OK ({len(required)} arquivos)")
    return True


def main():
    print("\n[ aula06-finetune · smoke test standalone ]\n")
    results = [
        test_imports(),
        test_env_vars(),
        test_prompts_file(),
        test_data_files(),
        test_openai_api(),
    ]
    print()
    if all(results):
        print("Tudo OK. O notebook do Colab deve rodar sem problemas neste setup.\n")
        return 0
    print("Pelo menos um teste falhou. Resolva antes de aplicar a aula.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
