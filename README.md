# aula06-finetune

Fine-tuning QLoRA do Qwen 2.5 7B Instruct usando Google Colab Free + OpenAI API.
Aula 06 — Engenharia de IA · Customização e Fine-Tuning de Modelos.

> **Pré-requisito conceitual**: Aula 04 finalizada (tag `aula04-final`) com
> golden dataset, rubrica e Model Card v1. A Aula 05 (tag `aula05-final`) é
> fortemente recomendada.
>
> **Pré-requisito prático**: NENHUM. Tudo que você precisa das aulas
> anteriores já está versionado em [`data/`](data/) — golden datasets,
> rubricas e evals baseline dos 2 produtos. Basta clonar e rodar.

## Stack simplificada

Uma única chave (OpenAI) cobre toda a aula:

- **GPU**: Google Colab Free (Tesla T4 16GB)
- **Modelo base**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (sem gating)
- **Framework**: Unsloth + TRL + PEFT
- **Teacher (síntese de dataset)**: OpenAI `gpt-4.1-mini`
- **Juiz (eval pós-fine-tune)**: OpenAI `gpt-4.1`
- **Storage do adapter**: Google Drive

## Custo total estimado por aluno

| Item | Modelo | Custo |
|---|---|---|
| Smoke test inicial | gpt-4.1-nano | < $0.001 |
| Síntese de 500 exemplos | gpt-4.1-mini | ~$0.70 |
| Eval de 5 queries com juiz | gpt-4.1 | ~$0.30 |
| Colab GPU | Free tier | $0 |
| **Total por aluno** | | **~$1.00** |

Recomendamos saldo mínimo de **USD 5** na conta OpenAI do aluno para cobrir
re-execuções e variações.

## O que já vem pronto no repo

Tudo que veio das Aulas 04/05 já está versionado — **você não precisa importar
nem baixar nada de outros repos**. O diretório `data/` contém, para ambos os
produtos:

| Arquivo | Origem | Função na Aula 06 |
|---|---|---|
| `golden_educiacao.csv` / `golden_designmind.csv` | Aula 04 | **Seeds** da síntese (50 prompts cada) |
| `rubric_educiacao.yaml` / `rubric_designmind.yaml` | Aula 04 | Rubrica do juiz LLM |
| `eval_baseline_educiacao.csv` / `eval_baseline_designmind.csv` | Aula 04 | Notas baseline pré-fine-tune (para o delta) |

Os dois produtos suportados são:

- **EducIAção** — tutor pedagógico para alunos em vulnerabilidade
- **DesignMind AI** — multi-agente de Design Thinking para PMEs

Escolha de produto: feita em UMA célula do notebook (Fase 2.1) ou via flag
`--seeds`/`--rubric`/`--baseline` nos scripts. Veja os comandos prontos abaixo.

---

## 1. Setup local (fora do Colab)

Requer **Python 3.10+** e uma `OPENAI_API_KEY` válida.

### Windows (PowerShell)

```powershell
# Clonar
git clone <url-do-repo>
cd aula06-finetune

# Criar e ativar venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Se bloquear por política de execução, rode uma única vez:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Dependências
python -m pip install --upgrade pip
pip install -r requirements.txt

# Variáveis de ambiente
Copy-Item .env.example .env
notepad .env   # preencha OPENAI_API_KEY=sk-proj-...

# Smoke test (valida imports + chave OpenAI)
python tests/test_smoke_colab.py
```

### Linux / macOS (bash/zsh)

```bash
# Clonar
git clone <url-do-repo>
cd aula06-finetune

# Criar e ativar venv
python3 -m venv .venv
source .venv/bin/activate

# Dependências
python -m pip install --upgrade pip
pip install -r requirements.txt

# Variáveis de ambiente
cp .env.example .env
$EDITOR .env   # preencha OPENAI_API_KEY=sk-proj-...

# Smoke test (valida imports + chave OpenAI)
python tests/test_smoke_colab.py
```

---

## 2. Passo a passo completo do exercício

O exercício inteiro roda dentro de `notebooks/aula06_finetune.ipynb`, em
**7 fases** (51 células). Tempo total: **~3h10**. Você pode rodar tudo no
Colab (caminho da aula) ou usar os scripts standalone como alternativa
(seção 3).

### Pré-Fase — Abrir o notebook no Colab (2 min)

1. Clone o repo no GitHub (ou abra direto pelo link que o professor enviar).
2. Acesse `notebooks/aula06_finetune.ipynb` no GitHub.
3. Clique no badge **"Open in Colab"** (ou copie a URL e abra em
   `colab.research.google.com → File → Open notebook → GitHub`).
4. No Colab: **Runtime → Change runtime type → T4 GPU → Save**.
5. No Colab: ícone de **chave** (Secrets) no painel esquerdo →
   adicione `OPENAI_API_KEY` com o valor `sk-proj-...` da sua conta.

> A partir daqui, execute as células em ordem. Cada subseção abaixo é uma
> fase do notebook.

---

### Fase 0 — Setup (7 min)

**O que acontece**: GPU, Drive e Secret são preparados.

**Células-chave**:
- `0.1` — `!nvidia-smi` → confirma Tesla T4 alocada
- `0.2` — monta Google Drive em `/content/drive`
- `0.3` — carrega `OPENAI_API_KEY` do Secret
- `0.4` — smoke test: chama `gpt-4.1-nano` com prompt `"diga apenas: pong"`

**Esperado**: imprime `"OpenAI OK: pong"` na célula 0.4. Custo < $0.001.

**Se falhar**:
- Sem GPU → Runtime → Change runtime type → T4 GPU.
- `OPENAI_API_KEY ausente` → adicione o Secret no painel de chave.

---

### Fase 1 — Carregar modelo e LoRA (8 min)

**O que acontece**: download de ~5GB do Qwen 2.5 7B Instruct pré-quantizado
em 4-bit; LoRA adapters são anexados em cima dele.

**Células-chave**:
- `1.1` — `pip install` de unsloth, trl, peft, accelerate, bitsandbytes.
  **⚠ Após esta célula: Runtime → Restart runtime, depois re-execute 0.2
  e 0.3** (sem restart, a 1.2 dá ImportError críptico).
- `1.2` — `FastLanguageModel.from_pretrained(...)` carrega Qwen em 4-bit
- `1.3` — `get_peft_model(...)` com `r=16, alpha=16`, target em todos os
  módulos Q/K/V/O e gate/up/down

**Esperado**: imprime `trainable% ≈ 1.05%`. Memória GPU ocupada ~6-8 GB.

**Se falhar**:
- `CUDA out of memory` → reinicie runtime e tente de novo (talvez sobrou
  estado de uma sessão anterior).

---

### Fase 2 — Sintetizar dataset (15 min, custo ~$0.70)

**O que acontece**: o teacher (`gpt-4.1-mini`) gera 10 variações para cada
um dos 50 prompts seed → **~500 exemplos** salvos em
`/content/drive/MyDrive/aula06_finetune/dataset_raw.jsonl`.

**Células-chave**:
- `2.1` — **VOCÊ ESCOLHE O PRODUTO AQUI**:
  ```python
  PRODUTO = "educiacao"   # ou "designmind"
  PRODUTO_NOME = "EducIAção (tutor pedagógico para alunos em vulnerabilidade)"
  ```
- `2.2` — define `SYNTHESIS_TEMPLATE` (mesmo template de `prompts/synthesis_template.txt`)
- `2.3` — loop sobre os 50 seeds, 10 variações cada, escreve `.jsonl` linha a linha

**Esperado**: barra de progresso `tqdm` mostrando ~500 chamadas. No fim:
`✓ Gerados: 500 exemplos / Custo estimado: ~USD 0.70`.

**Se falhar**:
- Rate limit OpenAI → aumente `time.sleep(0.05)` para `0.2` na célula 2.3.
- `golden_educiacao.csv não encontrado` → o notebook precisa que o repo
  esteja clonado em `/content/aula06-finetune/`. Verifique a célula de
  clone (deve aparecer logo após a 0.x).

---

### Fase 3 — Limpar e formatar (10 min)

**O que acontece**: filtros de qualidade aplicados ao raw → split 90/10
treino/teste → chat template do Qwen aplicado.

**Células-chave**:
- `3.1` — rejeita: campos vazios, `output` < 40 chars, duplicatas (hash MD5
  do input). Imprime distribuição por categoria.
- `3.2` — aplica `tokenizer.apply_chat_template(...)` e cria
  `Dataset.from_list(...).train_test_split(test_size=0.1, seed=42)`.

**Esperado**: `Limpos: ~450/500 (90%) / Treino: ~405, Teste: ~45`. Olhe a
amostra impressa para confirmar que o chat template inclui as tags
`<|im_start|>user / <|im_start|>assistant`.

---

### Fase 4 — Configurar trainer (5 min)

**O que acontece**: configura `SFTTrainer` da TRL com hiperparâmetros
calibrados para Qwen 7B + T4 + ~400 exemplos.

**Hiperparâmetros principais** (célula 4.1):
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`
  → batch efetivo = 8
- `num_train_epochs=2`, `learning_rate=2e-4`
- `optim="adamw_8bit"`, `lr_scheduler_type="linear"`

**Esperado**: imprime `Steps estimados: ~100 / Tempo estimado em T4: ~50 min`.

---

### Fase 5 — Training (60 min — teoria em paralelo)

**O que acontece**: training propriamente dito. Enquanto roda, a aula cobre
teoria (Phi-4, distillation, DPO/ORPO/KTO, debate PT-BR jurídico/médico).

**Célula-chave**: `5.1` — `trainer.train()`.

**Esperado**:
- Loss inicial: **2.5 - 3.0**
- Loss final: **0.8 - 1.2**
- Tempo total: **~50-60 min**

**Sinais de problema**:
- `Loss = 0.000` → overfitting catastrófico. **Pare imediatamente** e revise
  o dataset (provavelmente duplicatas escaparam do filtro 3.1).
- Loss travado em ~2.0+ → learning rate errado. Pare e reduza
  `learning_rate=2e-4` para `1e-4` na célula 4.1.
- Sessão Colab desconectou → o checkpoint está em `/content/outputs/`,
  você pode retomar.

---

### Fase 6 — Salvar adapter e eval contra golden (25 min, custo ~$0.30)

**O que acontece**: adapter salvo no Drive, modelo roda contra os 50 prompts
do golden, juiz (`gpt-4.1`) pontua cada resposta, comparação com baseline
da Aula 04 calcula o delta.

**Células-chave**:
- `6.1` — `model.save_pretrained(adapter_path)` → ~80-160 MB no Drive
- `6.2` — sanity check: gera 1 resposta para o primeiro prompt do golden
- `6.3` — loop sobre os 50 prompts → salva `eval_finetuned.csv`
- `6.4` — juiz LLM pontua cada resposta nas 5 dimensões da rubrica
- `6.5` — merge com baseline da Aula 04 → calcula delta e detecta
  catastrophic forgetting (delta < -0.3)

**Esperado** (output da célula 6.5):
```
=== COMPARAÇÃO ===
Baseline avg:   3.85
Fine-tuned avg: 4.10
Delta médio:    +0.25
✓ Sem catastrophic forgetting significativo
```

**Sinais de problema**:
- Delta negativo grande (< -0.2) → o fine-tune piorou o modelo. Provavelmente
  dataset muito enviesado ou epochs demais. Revise 3.1 e 4.1.
- Catastrophic forgetting em ≥2 queries → mesmo diagnóstico.

---

### Fase 7 — Model Card v3 e entrega (20 min)

**O que acontece**: gera relatório markdown com métricas finais e decisão
de promoção do adapter.

**Células-chave**:
- `7.1` — gera `model_card_v3_section.md` com as seções C (Pipeline),
  D (Dataset), E (Resultados), F (Decisão).

**Próximos passos (manuais, fora do notebook)**:

1. Baixe do Drive para o seu repo local:
   ```
   /content/drive/MyDrive/aula06_finetune/dataset_raw.jsonl
   /content/drive/MyDrive/aula06_finetune/eval_finetuned_vs_baseline.csv
   /content/drive/MyDrive/aula06_finetune/model_card_v3_section.md
   ```

2. Coloque cada um no lugar certo no repo local:
   ```bash
   cp ~/Downloads/dataset_raw.jsonl              data/dataset_sintetico.jsonl
   cp ~/Downloads/eval_finetuned_vs_baseline.csv results/
   ```

3. Crie `docs/model_card_v3.md` a partir do template:
   ```bash
   cp docs/model_card_v3_template.md docs/model_card_v3.md
   # cole o conteúdo de model_card_v3_section.md nas seções C-F
   # preencha as células com <!-- preencher -->
   ```

4. Commit + tag final:
   ```bash
   git checkout -b aula06-finetune
   git add data/dataset_sintetico.jsonl results/eval_finetuned_vs_baseline.csv docs/model_card_v3.md
   git commit -m "Aula 06: fine-tune QLoRA Qwen 2.5 7B — adapter v1"
   git tag aula06-final
   git push origin aula06-finetune --tags
   ```

5. **Apresente em sala** (3 min):
   - Produto escolhido e por que precisava de fine-tune (não bastaria prompt+RAG?)
   - Dataset: tamanho, qualidade, custo
   - Delta vs baseline + onde houve regressão
   - Trade-off encontrado + próximo passo

---

## 3. Pipeline standalone (fora do Colab) — comandos prontos

Os 3 scripts em `scripts/` reproduzem as fases de **síntese**, **avaliação com
juiz** e **comparação** sem precisar do Colab. Útil para regenerar dados,
rodar em CI ou debugar.

> **Vindo das Aulas 04/05?** Os scripts foram renomeados/reorganizados nesta
> Aula 06. Mapeamento:
>
> | Slides/material anterior | Equivalente nesta Aula 06 |
> |---|---|
> | `check_candidates.py` (validar candidatos antes da eval) | `python tests/test_smoke_colab.py` + filtros da Fase 3.1 do notebook |
> | `run_eval.py` (rodar avaliação completa) | `scripts/judge_eval.py` (ou Fases 6.3 + 6.4 do notebook) |
> | `analyze.py` (analisar resultados e gerar matriz) | `scripts/compare_baseline.py` (ou Fases 6.5 + 7.1 do notebook) |

Antes de qualquer comando: `.venv` ativo e `.env` configurado.

### 3.A — Produto EducIAção (copy-paste completo)

**Linux / macOS:**

```bash
# 1) Sintetizar dataset (~$0.70, ~10 min)
python scripts/synth_dataset.py \
  --seeds data/golden_educiacao.csv \
  --produto "EducIAção (tutor pedagógico para alunos em vulnerabilidade)" \
  --teacher gpt-4.1-mini \
  --n-per-seed 10 \
  --out data/dataset_raw_educiacao.jsonl

# 2) [Treine no Colab e exporte results/eval_finetuned_educiacao.csv]
#    Colunas esperadas: id, input, output_finetuned

# 3) Pontuar respostas com o juiz (~$0.30)
python scripts/judge_eval.py \
  --eval-csv results/eval_finetuned_educiacao.csv \
  --golden-csv data/golden_educiacao.csv \
  --rubric-yaml data/rubric_educiacao.yaml \
  --baseline-csv data/eval_baseline_educiacao.csv \
  --judge-model gpt-4.1 \
  --out results/eval_finetuned_scored_educiacao.csv

# 4) Relatório final baseline vs fine-tuned
python scripts/compare_baseline.py \
  --scored results/eval_finetuned_scored_educiacao.csv \
  --rubric data/rubric_educiacao.yaml \
  --out results/comparison_report_educiacao.md
```

**Windows (PowerShell):** mesma sequência, mas troque `\` no fim da linha por
backtick `` ` `` — ou mantenha o comando em uma única linha:

```powershell
python scripts/synth_dataset.py --seeds data/golden_educiacao.csv --produto "EducIAção (tutor pedagógico para alunos em vulnerabilidade)" --teacher gpt-4.1-mini --n-per-seed 10 --out data/dataset_raw_educiacao.jsonl

python scripts/judge_eval.py --eval-csv results/eval_finetuned_educiacao.csv --golden-csv data/golden_educiacao.csv --rubric-yaml data/rubric_educiacao.yaml --baseline-csv data/eval_baseline_educiacao.csv --judge-model gpt-4.1 --out results/eval_finetuned_scored_educiacao.csv

python scripts/compare_baseline.py --scored results/eval_finetuned_scored_educiacao.csv --rubric data/rubric_educiacao.yaml --out results/comparison_report_educiacao.md
```

### 3.B — Produto DesignMind AI (copy-paste completo)

**Linux / macOS:**

```bash
# 1) Sintetizar dataset
python scripts/synth_dataset.py \
  --seeds data/golden_designmind.csv \
  --produto "DesignMind AI (multi-agente de Design Thinking para PMEs)" \
  --teacher gpt-4.1-mini \
  --n-per-seed 10 \
  --out data/dataset_raw_designmind.jsonl

# 2) [Treine no Colab e exporte results/eval_finetuned_designmind.csv]

# 3) Pontuar respostas com o juiz
python scripts/judge_eval.py \
  --eval-csv results/eval_finetuned_designmind.csv \
  --golden-csv data/golden_designmind.csv \
  --rubric-yaml data/rubric_designmind.yaml \
  --baseline-csv data/eval_baseline_designmind.csv \
  --judge-model gpt-4.1 \
  --out results/eval_finetuned_scored_designmind.csv

# 4) Relatório final baseline vs fine-tuned
python scripts/compare_baseline.py \
  --scored results/eval_finetuned_scored_designmind.csv \
  --rubric data/rubric_designmind.yaml \
  --out results/comparison_report_designmind.md
```

**Windows (PowerShell):**

```powershell
python scripts/synth_dataset.py --seeds data/golden_designmind.csv --produto "DesignMind AI (multi-agente de Design Thinking para PMEs)" --teacher gpt-4.1-mini --n-per-seed 10 --out data/dataset_raw_designmind.jsonl

python scripts/judge_eval.py --eval-csv results/eval_finetuned_designmind.csv --golden-csv data/golden_designmind.csv --rubric-yaml data/rubric_designmind.yaml --baseline-csv data/eval_baseline_designmind.csv --judge-model gpt-4.1 --out results/eval_finetuned_scored_designmind.csv

python scripts/compare_baseline.py --scored results/eval_finetuned_scored_designmind.csv --rubric data/rubric_designmind.yaml --out results/comparison_report_designmind.md
```

### Parâmetros úteis dos scripts

| Script | Flag | Default | Notas |
|---|---|---|---|
| `synth_dataset.py` | `--n-per-seed` | 10 | 50 seeds × 10 = 500 exemplos (alinhado com o custo de $0.70). Aumente se quiser mais. |
| `synth_dataset.py` | `--tam-min` / `--tam-max` | 100 / 300 | Faixa de palavras das respostas |
| `synth_dataset.py` | `--sleep-ms` | 50 | Pausa entre chamadas (rate limit) |
| `synth_dataset.py` | `--teacher` | `gpt-4.1-mini` | Pode trocar por `gpt-4.1` (mais caro) |
| `judge_eval.py` | `--judge-model` | `gpt-4.1` | Não recomendamos modelos mais fracos como juiz |

---

## Dois casos de uso suportados (no notebook)

A escolha é feita na Fase 2.1 do notebook. Os arquivos correspondentes do
`data/` são lidos automaticamente.

### EducIAção (tutor pedagógico)
```python
PRODUTO = "educiacao"
PRODUTO_NOME = "EducIAção (tutor pedagógico para alunos em vulnerabilidade)"
```

### DesignMind AI (multi-agente de Design Thinking)
```python
PRODUTO = "designmind"
PRODUTO_NOME = "DesignMind AI (multi-agente de Design Thinking para PMEs)"
```

## Estrutura

```
aula06-finetune/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   └── aula06_finetune.ipynb       # notebook principal (7 fases, 51 células)
│
├── scripts/
│   ├── synth_dataset.py            # síntese standalone fora do Colab
│   ├── judge_eval.py               # pontua respostas usando rubrica
│   └── compare_baseline.py         # tabela baseline vs fine-tuned
│
├── prompts/
│   └── synthesis_template.txt      # template parametrizável
│
├── data/                            # JÁ VEM PRONTO — herdado das Aulas 04/05
│   ├── golden_educiacao.csv
│   ├── golden_designmind.csv
│   ├── rubric_educiacao.yaml
│   ├── rubric_designmind.yaml
│   ├── eval_baseline_educiacao.csv
│   └── eval_baseline_designmind.csv
│
├── docs/
│   └── model_card_v3_template.md   # template a preencher
│
├── tests/
│   ├── test_smoke_colab.py         # validação fora do Colab
│   └── build_notebook.py           # gerador do .ipynb (uso interno)
│
└── results/                         # outputs (não versionados)
```

## Entregas obrigatórias da aula

| # | Entrega | Onde fica | Como gerar |
|---|---|---|---|
| 1 | Branch `aula06-finetune` no repo principal | GitHub | `git checkout -b aula06-finetune` |
| 2 | Adapter QLoRA treinado | `/content/drive/MyDrive/aula06_finetune/adapter_v1/` (Drive) | Saída da Fase 6.1 |
| 3 | Dataset sintético versionado (300+ exemplos) | `data/dataset_sintetico.jsonl` | Saída da Fase 2.3, baixe do Drive |
| 4 | Comparação baseline vs fine-tuned | `results/eval_finetuned_vs_baseline.csv` | Saída da Fase 6.5, baixe do Drive |
| 5 | Model Card v3 preenchido | `docs/model_card_v3.md` | Use o template + relatório da Fase 7.1 |
| 6 | Tag `aula06-final` | GitHub | `git tag aula06-final && git push --tags` |

Checklist de entrega final:

```bash
# Antes de aplicar a tag, confirme que tudo está no lugar:
ls data/dataset_sintetico.jsonl                    # entrega 3
ls results/eval_finetuned_vs_baseline.csv          # entrega 4
ls docs/model_card_v3.md                           # entrega 5

# Verifique que o adapter está no Drive (não precisa estar no git):
ls /content/drive/MyDrive/aula06_finetune/adapter_v1/  # entrega 2

# Commit, push, tag:
git add data/dataset_sintetico.jsonl results/eval_finetuned_vs_baseline.csv docs/model_card_v3.md
git commit -m "Aula 06: fine-tune QLoRA Qwen 2.5 7B — adapter v1"
git push origin aula06-finetune
git tag aula06-final
git push origin --tags
```

## Troubleshooting rápido

- **`OPENAI_API_KEY não encontrada`** → faltou `cp .env.example .env` ou
  exportar a variável no ambiente atual (`source .env` ou reabra o terminal).
- **`ModuleNotFoundError: openai`** → `.venv` não está ativo, ou
  `pip install -r requirements.txt` ainda não foi rodado.
- **`Set-ExecutionPolicy` bloqueado no Windows** → rode o PowerShell como
  administrador uma vez, ou use o comando da seção de Setup local.
- **Custo acima do esperado** → reduza `--n-per-seed` no `synth_dataset.py`,
  ou troque `--teacher gpt-4.1-mini` por `gpt-4.1-nano` para um smoke run.
