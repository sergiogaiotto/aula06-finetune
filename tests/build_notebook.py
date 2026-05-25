"""Gera o notebook aula06_finetune.ipynb com todas as 7 fases do protocolo.
Versão OpenAI (gpt-4.1-mini teacher + gpt-4.1 juiz).
Roda uma vez localmente para produzir o .ipynb final.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "notebooks" / "aula06_finetune.ipynb"


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")],
    }


CELLS = []

# ===== HEADER =====
CELLS.append(md_cell("""# Aula 06 — Fine-Tuning QLoRA com Qwen 2.5 7B

> Engenharia de IA · Sessão 6 de 8
> Continuação direta das Aulas 04 e 05.
> **Stack OpenAI**: gpt-4.1-mini (teacher) + gpt-4.1 (juiz). Custo total ~USD 1 por aluno.

Este notebook executa o protocolo prático completo da Aula 06 em 7 fases:

| Fase | Objetivo | Tempo |
|---|---|---|
| 0 | Setup Colab + GPU + Drive + Secret | 7 min |
| 1 | Carregar Qwen 2.5 7B + anexar LoRA | 8 min |
| 2 | Sintetizar dataset via teacher (OpenAI) | 15 min |
| 3 | Limpar e formatar com chat template | 10 min |
| 4 | Configurar SFTTrainer | 5 min |
| 5 | Training (60 min, teoria em paralelo) | 60 min |
| 6 | Salvar adapter + eval contra golden | 25 min |
| 7 | Model Card v3 + defesa em sala | 20 min |

**Total**: ~3h10

## Pré-requisitos

Antes de começar, suba estes arquivos no Colab (painel esquerdo → ícone de pasta → upload):
1. `golden_<produto>.csv` (do repositório aula06-finetune/data/)
2. `rubric_<produto>.yaml` (do repositório aula06-finetune/data/)
3. `eval_baseline_<produto>.csv` (do repositório aula06-finetune/data/)

E configure o Secret (ícone de chave no painel esquerdo):
- `OPENAI_API_KEY` com saldo mínimo de USD 5
"""))

# ===== FASE 0 =====
CELLS.append(md_cell("""## Fase 0 — Setup (7 min)

Garantir GPU alocada, Drive montado e Secret carregado ANTES de qualquer outra coisa.
"""))

CELLS.append(md_cell("### 0.1 — Verificar GPU"))
CELLS.append(code_cell("""!nvidia-smi"""))

CELLS.append(md_cell("""Esperado: aparecer "Tesla T4" com ~15GB de memória.

Se aparecer "command not found", vá em **Runtime → Change runtime type → T4 GPU → Save** e re-execute a célula."""))

CELLS.append(md_cell("### 0.2 — Montar Google Drive"))
CELLS.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
WORK_DIR = '/content/drive/MyDrive/aula06_finetune'
os.makedirs(WORK_DIR, exist_ok=True)
print(f"Pasta de trabalho: {WORK_DIR}")"""))

CELLS.append(md_cell("### 0.3 — Carregar Secret OpenAI"))
CELLS.append(code_cell("""from google.colab import userdata
import os

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Validação rápida (sem imprimir a chave)
assert os.environ['OPENAI_API_KEY'].startswith('sk-'), "OPENAI_API_KEY ausente ou inválida"
print("Secret OpenAI carregado ✓")"""))

CELLS.append(md_cell("### 0.4 — Teste rápido de conexão OpenAI"))
CELLS.append(code_cell("""# Smoke test com gpt-4.1-nano (modelo mais barato, < $0.001)
from openai import OpenAI

oai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
r = oai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "diga apenas: pong"}],
    max_tokens=10,
)
print(f"OpenAI OK: {r.choices[0].message.content}")"""))

# ===== FASE 1 =====
CELLS.append(md_cell("""## Fase 1 — Carregar modelo e LoRA (8 min)

Download de ~5GB do Qwen 2.5 7B Instruct pré-quantizado em 4-bit. Anexa adapters LoRA.
Apenas ~1% dos parâmetros serão treinados.

**IMPORTANTE**: após a célula 1.1 instalar dependências, vá em **Runtime → Restart runtime** e re-execute Fases 0.2 e 0.3 antes de prosseguir."""))

CELLS.append(md_cell("### 1.1 — Instalar dependências"))
CELLS.append(code_cell("""!pip install -q --upgrade unsloth trl peft accelerate bitsandbytes datasets openai

# IMPORTANTE: depois desta célula, Runtime → Restart runtime
# Sem restart, dá ImportError críptico na célula 1.2.
# Após restart, re-execute as células 0.2 (Drive) e 0.3 (Secret)."""))

CELLS.append(md_cell("### 1.2 — Carregar Qwen 2.5 7B em 4-bit"))
CELLS.append(code_cell("""from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo carregado: {n_params:,} parâmetros (~{n_params/1e9:.1f}B)")"""))

CELLS.append(md_cell("### 1.3 — Anexar adapters LoRA"))
CELLS.append(code_cell("""model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()
# Esperado: trainable% ≈ 1.05%"""))

# ===== FASE 2 =====
CELLS.append(md_cell("""## Fase 2 — Sintetizar dataset (15 min)

Usa OpenAI gpt-4.1-mini como teacher para gerar 10 variações de cada um dos 50 prompts do golden dataset (~500 exemplos totais).

**Pré-requisito**: arquivos do produto já subidos no Colab.

## Escolha o produto: descomente APENAS UMA das duas opções abaixo."""))

CELLS.append(md_cell("### 2.1 — Configurar variáveis do produto"))
CELLS.append(code_cell("""# ============================================
# ESCOLHA SEU PRODUTO — descomente UMA opção
# ============================================

# Opção A · EducIAção
PRODUTO = "educiacao"
PRODUTO_NOME = "EducIAção (tutor pedagógico para alunos em vulnerabilidade)"

# Opção B · DesignMind AI (descomente as 2 linhas abaixo se for este produto)
# PRODUTO = "designmind"
# PRODUTO_NOME = "DesignMind AI (multi-agente de Design Thinking para PMEs)"

# ============================================
# Configuração técnica (mantenha)
# ============================================
SEEDS_CSV = f"golden_{PRODUTO}.csv"
RUBRIC_YAML = f"rubric_{PRODUTO}.yaml"
BASELINE_CSV = f"eval_baseline_{PRODUTO}.csv"

TEACHER_MODEL = "gpt-4.1-mini"  # síntese: cheap, JSON mode, 1M context
JUDGE_MODEL = "gpt-4.1"          # juiz: frontier quality
N_PER_SEED = 10                  # variações por seed (50 seeds × 10 = 500 exemplos)
TAM_MIN, TAM_MAX = 100, 300      # comprimento da resposta esperado

print(f"Produto:  {PRODUTO_NOME}")
print(f"Teacher:  {TEACHER_MODEL}")
print(f"Juiz:     {JUDGE_MODEL}")
print(f"Seeds:    {SEEDS_CSV}")
print(f"Rubrica:  {RUBRIC_YAML}")
print(f"Baseline: {BASELINE_CSV}")"""))

CELLS.append(md_cell("### 2.2 — Definir o prompt de síntese"))
CELLS.append(code_cell("""SYNTHESIS_TEMPLATE = '''Você é um gerador especializado de exemplos de treinamento para fine-tuning.

A partir do PROMPT ORIGINAL abaixo, gere UMA variação semanticamente equivalente — mesma intenção, mesma categoria, mesmo nível de dificuldade — porém com texto, persona, contexto ou exemplo deliberadamente diferentes do original.

Em seguida, gere a RESPOSTA IDEAL para essa variação, seguindo estritamente o PADRÃO DE RESPOSTA ESPERADA fornecido.

REGRAS DURAS:
1. A variação NÃO pode ser apenas reformulação superficial.
2. A resposta IDEAL deve evitar TODAS as RED FLAGS listadas.
3. Tom alinhado ao produto: {produto}.
4. Tamanho da resposta entre {tam_min} e {tam_max} palavras.
5. Português brasileiro natural, sem clichês ou frases motivacionais.

PROMPT ORIGINAL:
{original}

PADRÃO ESPERADO:
{expected}

RED FLAGS:
{red_flags}

Retorne APENAS um JSON válido:
{{"input": "<variação>", "output": "<resposta ideal>"}}
'''
print("Template carregado.")
print(SYNTHESIS_TEMPLATE[:300] + "...")"""))

CELLS.append(md_cell("### 2.3 — Executar a síntese"))
CELLS.append(code_cell("""from openai import OpenAI
import pandas as pd
import json
import time
from tqdm.notebook import tqdm

oai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

seeds = pd.read_csv(SEEDS_CSV)
print(f"Carregados {len(seeds)} seeds")

dataset = []
out_path = f"{WORK_DIR}/dataset_raw.jsonl"

with open(out_path, 'w', encoding='utf-8') as f:
    for _, row in seeds.iterrows():
        print(f"\\n  expandindo {row['id']} ({row.get('category', 'sem categoria')})...")
        for i in tqdm(range(N_PER_SEED), desc=f"  {row['id']}"):
            try:
                resp = oai.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "user", "content": SYNTHESIS_TEMPLATE.format(
                        produto=PRODUTO_NOME,
                        tam_min=TAM_MIN, tam_max=TAM_MAX,
                        original=row['input'],
                        expected=row['expected_pattern'],
                        red_flags=row['red_flags'],
                    )}],
                    response_format={"type": "json_object"},
                    max_tokens=800,
                    temperature=0.8,
                )
                ex = json.loads(resp.choices[0].message.content)
                if ex.get('input') and ex.get('output'):
                    ex['source_seed'] = row['id']
                    ex['category'] = row.get('category', '')
                    f.write(json.dumps(ex, ensure_ascii=False) + '\\n')
                    dataset.append(ex)
            except Exception as e:
                print(f"    [warn] falha: {str(e)[:80]}")
            time.sleep(0.05)  # OpenAI rate limit é generoso

print(f"\\n✓ Gerados: {len(dataset)} exemplos")
print(f"  Salvos em: {out_path}")
print(f"  Custo estimado: ~USD {len(dataset) * 0.0014:.2f}")"""))

# ===== FASE 3 =====
CELLS.append(md_cell("""## Fase 3 — Limpar e formatar (10 min)

Filtros de qualidade: respostas curtas demais, duplicatas, entradas vazias.
Em seguida, aplica o chat template do Qwen."""))

CELLS.append(md_cell("### 3.1 — Filtros de qualidade"))
CELLS.append(code_cell("""import hashlib

raw = [json.loads(line) for line in open(f"{WORK_DIR}/dataset_raw.jsonl")]
print(f"Brutos: {len(raw)}")

seen_hashes = set()
clean = []
rejected_by_reason = {"missing": 0, "too_short": 0, "duplicate": 0}

for ex in raw:
    if not ex.get('input') or not ex.get('output'):
        rejected_by_reason["missing"] += 1
        continue
    if len(ex['output'].strip()) < 40:
        rejected_by_reason["too_short"] += 1
        continue
    h = hashlib.md5(ex['input'].lower().strip().encode()).hexdigest()
    if h in seen_hashes:
        rejected_by_reason["duplicate"] += 1
        continue
    seen_hashes.add(h)
    clean.append(ex)

print(f"Limpos: {len(clean)}/{len(raw)} ({len(clean)/len(raw)*100:.0f}%)")
print(f"Rejeições: {rejected_by_reason}")

# Distribuição por categoria
from collections import Counter
dist = Counter(ex['category'] for ex in clean)
print(f"\\nDistribuição por categoria:")
for cat, n in dist.most_common():
    print(f"  {cat:30s} {n}")"""))

CELLS.append(md_cell("### 3.2 — Aplicar chat template do Qwen"))
CELLS.append(code_cell("""from datasets import Dataset

def format_example(ex):
    msgs = [
        {"role": "user",      "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )}

ds = Dataset.from_list(clean).map(format_example)
ds = ds.train_test_split(test_size=0.1, seed=42)
print(f"Treino: {len(ds['train'])}, Teste: {len(ds['test'])}")

print("\\n----- Amostra do dataset formatado -----")
print(ds['train'][0]['text'][:500])
print("\\n----- /Amostra -----")"""))

# ===== FASE 4 =====
CELLS.append(md_cell("""## Fase 4 — Configurar trainer (5 min)

Hiperparâmetros calibrados para Qwen 7B + T4 + ~400 exemplos."""))

CELLS.append(md_cell("### 4.1 — Configurar SFTTrainer"))
CELLS.append(code_cell("""from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    args=SFTConfig(
        output_dir="/content/outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
    ),
    dataset_text_field="text",
    max_seq_length=2048,
)
print("Trainer configurado.")
print(f"Steps estimados: {len(ds['train']) * 2 // (2 * 4)}")
print(f"Tempo estimado em T4: ~{len(ds['train']) * 2 // (2 * 4) * 30 / 60:.0f} min")"""))

# ===== FASE 5 =====
CELLS.append(md_cell("""## Fase 5 — Training (60 min — teoria em paralelo)

Loss inicial ~2.5-3.0, final ~0.8-1.2.

**Enquanto roda, a turma cobre:**
- Estudo de caso Phi-4 (textbook revolution)
- Distillation: DeepSeek R1 → variantes destiladas
- ORPO/DPO/KTO: quando usar cada um
- Debate: fine-tuning em PT-BR jurídico/médico

**Sinais de problema:**
- Loss = 0.000: overfitting catastrófico. Parar.
- Loss travado: learning rate errado. Parar e reduzir para 1e-4.
- Sessão desconectou: perda parcial. Há checkpoint salvo em /content/outputs/."""))

CELLS.append(md_cell("### 5.1 — Iniciar training (não interromper)"))
CELLS.append(code_cell("""trainer_stats = trainer.train()

print(f"\\n✓ Training concluído")
print(f"  Loss final: {trainer_stats.training_loss:.4f}")
print(f"  Tempo total: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")"""))

# ===== FASE 6 =====
CELLS.append(md_cell("""## Fase 6 — Salvar adapter e eval contra golden (25 min)

**Fase obrigatória**. Sem ela, fine-tuning é fé."""))

CELLS.append(md_cell("### 6.1 — Salvar adapter no Drive"))
CELLS.append(code_cell("""adapter_path = f"{WORK_DIR}/adapter_v1"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Adapter salvo em {adapter_path}")
!du -sh {adapter_path}
# Esperado: ~80-160 MB"""))

CELLS.append(md_cell("### 6.2 — Sanity check de inferência"))
CELLS.append(code_cell("""FastLanguageModel.for_inference(model)

def generate(prompt, max_new_tokens=400):
    msgs = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )

# Teste com 1 prompt do golden
sample_prompt = seeds.iloc[0]['input']
print("INPUT:", sample_prompt[:200])
print("\\nOUTPUT FINE-TUNED:")
print(generate(sample_prompt))"""))

CELLS.append(md_cell("### 6.3 — Rodar todo o golden contra o modelo fine-tuned"))
CELLS.append(code_cell("""results = []
for _, row in seeds.iterrows():
    print(f"  rodando {row['id']}...", end=' ')
    output = generate(row['input'], max_new_tokens=600)
    results.append({
        'id': row['id'],
        'input': row['input'],
        'output_finetuned': output,
        'category': row.get('category', ''),
    })
    print('✓')

eval_df = pd.DataFrame(results)
eval_path = f"{WORK_DIR}/eval_finetuned.csv"
eval_df.to_csv(eval_path, index=False)
print(f"\\n✓ Salvo em {eval_path}")"""))

CELLS.append(md_cell("### 6.4 — Pontuar com o juiz (mesma rubrica das Aulas 04/05)"))
CELLS.append(code_cell("""import yaml
import textwrap

rubric = yaml.safe_load(open(RUBRIC_YAML))

def build_judge_prompt(user_input, candidate_output, expected_pattern, red_flags, rubric):
    dims_block = []
    for dim in rubric["dimensions"]:
        scale = "\\n".join(f"      {k}: {v}" for k, v in dim["scale"].items())
        dims_block.append(
            f"  - id: {dim['name']}\\n"
            f"    label: {dim['label']}\\n"
            f"    weight: {dim['weight']}\\n"
            f"    description: {dim['description']}\\n"
            f"    scale:\\n{scale}"
        )
    dims_yaml = "\\n".join(dims_block)
    schema = {dim["name"]: "<int 1-5>" for dim in rubric["dimensions"]}
    schema["red_flags_acionadas"] = ["<string>"]
    schema["justificativa_curta"] = "<string até 80 palavras>"
    return textwrap.dedent(f'''
    Você é um juiz especialista. Avalie a RESPOSTA do modelo fine-tuned
    contra a RUBRICA com dimensões ponderadas.

    REGRAS: Avalie pelas âncoras 1-5. NÃO premie comprimento ou formatação rica.
    Red flags forçam nota 1-2. Saída APENAS JSON, sem markdown wrapper.

    INPUT: {user_input}

    RESPOSTA: {candidate_output}

    PADRÃO ESPERADO: {expected_pattern}

    RED FLAGS: {red_flags}

    RUBRICA:
    {dims_yaml}

    SCHEMA: {json.dumps(schema, indent=2, ensure_ascii=False)}

    Retorne apenas o JSON.
    ''').strip()

scored_rows = []
for _, r in tqdm(eval_df.iterrows(), total=len(eval_df), desc="  julgando"):
    seed_row = seeds[seeds['id'] == r['id']].iloc[0]
    prompt = build_judge_prompt(
        r['input'], r['output_finetuned'],
        seed_row['expected_pattern'], seed_row['red_flags'],
        rubric,
    )
    try:
        resp = oai.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        weighted = sum(
            float(scores.get(d['name'], 0)) * float(d['weight'])
            for d in rubric['dimensions']
        )
        row_out = {'id': r['id'], 'weighted_finetuned': round(weighted, 3)}
        for d in rubric['dimensions']:
            row_out[f"{d['name']}_finetuned"] = scores.get(d['name'])
        scored_rows.append(row_out)
    except Exception as e:
        print(f"    [warn] {r['id']}: {str(e)[:80]}")
        scored_rows.append({'id': r['id'], 'weighted_finetuned': None, 'error': str(e)})

scored_df = pd.DataFrame(scored_rows)
scored_df.to_csv(f"{WORK_DIR}/eval_finetuned_scored.csv", index=False)
print(f"\\n  Média ponderada fine-tuned: {scored_df['weighted_finetuned'].mean():.3f}")"""))

CELLS.append(md_cell("### 6.5 — Comparar com baseline da Aula 04"))
CELLS.append(code_cell("""try:
    baseline_raw = pd.read_csv(BASELINE_CSV)
    if 'model' in baseline_raw.columns:
        baseline = baseline_raw.groupby('id')['weighted'].mean().rename('weighted_baseline')
    else:
        baseline = baseline_raw.set_index('id')['weighted'].rename('weighted_baseline')

    final = scored_df.merge(baseline, left_on='id', right_index=True, how='left')
    final['delta'] = final['weighted_finetuned'] - final['weighted_baseline']
    final.to_csv(f"{WORK_DIR}/eval_finetuned_vs_baseline.csv", index=False)

    print(f"\\n  === COMPARAÇÃO ===")
    print(f"  Baseline avg:   {final['weighted_baseline'].mean():.3f}")
    print(f"  Fine-tuned avg: {final['weighted_finetuned'].mean():.3f}")
    delta_avg = (final['weighted_finetuned'] - final['weighted_baseline']).mean()
    print(f"  Delta médio:    {delta_avg:+.3f}")

    # Catastrophic forgetting check
    regression = final[final['delta'] < -0.3]
    if len(regression) > 0:
        print(f"\\n  ⚠ CATASTROPHIC FORGETTING em {len(regression)} queries:")
        for _, row in regression.iterrows():
            print(f"    {row['id']}: {row['delta']:+.3f}")
    else:
        print("\\n  ✓ Sem catastrophic forgetting significativo")

    display(final.sort_values('delta', ascending=False))
except FileNotFoundError:
    print(f"  [warn] {BASELINE_CSV} não encontrado.")
    print(f"  Suba o CSV de baseline e re-execute esta célula.")"""))

# ===== FASE 7 =====
CELLS.append(md_cell("""## Fase 7 — Model Card v3 e defesa (20 min)

Última fase. Incrementa o Model Card v2 da Aula 05 com as seções de fine-tuning."""))

CELLS.append(md_cell("### 7.1 — Gerar relatório markdown para colar no Model Card"))
CELLS.append(code_cell("""from datetime import datetime

avg_ft = scored_df['weighted_finetuned'].mean()
n_examples = len(clean)
adapter_size_mb = round(os.path.getsize(f"{adapter_path}/adapter_model.safetensors") / 1e6, 1)

if 'final' in dir():
    avg_bl = final['weighted_baseline'].mean()
    delta = avg_ft - avg_bl
    forgetting_n = len(final[final['delta'] < -0.3])
else:
    avg_bl, delta, forgetting_n = None, None, None

report = f'''## Model Card v3 — Pipeline de fine-tuning

### C · Pipeline
| Campo | Valor |
|---|---|
| Modelo base | unsloth/Qwen2.5-7B-Instruct-bnb-4bit |
| Método | QLoRA (r=16, alpha=16) |
| Hardware | Google Colab Free (Tesla T4 16GB) |
| Tempo training | {trainer_stats.metrics.get('train_runtime', 0):.0f}s |
| Tamanho adapter | {adapter_size_mb} MB |

### D · Dataset sintético
| Campo | Valor |
|---|---|
| Teacher | {TEACHER_MODEL} (OpenAI) |
| Total gerados | {len(raw)} |
| Pós-filtragem | {n_examples} |
| Custo síntese | ~USD {len(raw) * 0.0014:.2f} |
| Data | {datetime.now().isoformat()} |

### E · Resultados pós-eval
| Métrica | Valor |
|---|---|
| Juiz | {JUDGE_MODEL} (OpenAI) |
| Score baseline (Aula 04) | {f"{avg_bl:.3f}" if avg_bl else "—"} |
| Score fine-tuned | {avg_ft:.3f} |
| Delta | {f"{delta:+.3f}" if delta is not None else "—"} |
| Catastrophic forgetting | {forgetting_n if forgetting_n is not None else "—"} queries |

### F · Decisão
- **Promover o adapter?** {'SIM' if delta and delta > 0.2 else 'NÃO' if delta and delta < 0 else 'REVISAR MANUALMENTE'}
- **Justificativa**: <preencher>
- **O que faria diferente em uma v2**: <preencher>
'''
print(report)

with open(f"{WORK_DIR}/model_card_v3_section.md", 'w') as f:
    f.write(report)
print(f"\\n✓ Relatório salvo em {WORK_DIR}/model_card_v3_section.md")"""))

CELLS.append(md_cell("""### 7.2 — Próximos passos manuais (fora do notebook)

1. Baixe `model_card_v3_section.md` do Drive e cole nas seções C-F do seu `docs/model_card_v2.md`, renomeando-o para `model_card_v3.md`.
2. Faça commit do repositório local com os novos arquivos:
   - `data/dataset_sintetico.jsonl` (do Drive)
   - `results/eval_finetuned_vs_baseline.csv` (do Drive)
   - `docs/model_card_v3.md`
3. Aplique a tag:
   ```bash
   git tag aula06-final
   git push origin aula06-finetune --tags
   ```
4. Apresente em sala (3 min): produto, dataset, delta vs baseline, trade-off, próximo passo.

## Continuidade

O adapter salvo em `/content/drive/MyDrive/aula06_finetune/adapter_v1/` será carregado nas próximas aulas:

- **Aula 07 (Agentes)**: agente usa o adapter como gerador, com tools (incluindo RAG da Aula 05).
- **Aula 08 (LLMOps)**: adapter passa por gates de regressão no CI/CD.
"""))

# Monta o notebook
nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Notebook gerado: {OUT}")
print(f"Tamanho: {OUT.stat().st_size / 1024:.1f} KB")
print(f"Total de células: {len(CELLS)}")
