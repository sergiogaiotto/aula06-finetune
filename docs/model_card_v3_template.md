# Model Card v3 · com camada de fine-tuning

> Incrementa o Model Card v2 da Aula 05. NÃO substitui — adiciona seções
> C, D, E e F sobre o pipeline de fine-tuning construído na Aula 06.
> Mantenha intactas as seções da v1 (Aula 04) e v2 (Aula 05).

---

## A) Continuidade (da v1 e v2 — não reescrever)

| Campo | Valor |
|---|---|
| Modelo gerador (campeão Aula 04) | <!-- preencher --> |
| Pipeline RAG (Aula 05) | <!-- modo escolhido --> |
| Rubrica original (5 dimensões) | mantida |
| Tag v1 | aula04-final |
| Tag v2 | aula05-final |

---

## B) Pipeline RAG (mantida da v2 — não reescrever)

(seções da v2 inalteradas)

---

## C) Pipeline de Fine-Tuning (novo · Aula 06)

### C.1 Decisão
| Campo | Valor |
|---|---|
| Por que fine-tuning? | <!-- estilo? formato? domínio? destilar para menor? --> |
| Por que NÃO bastaria prompt+RAG? | <!-- justificar com evidência da Aula 05 --> |

### C.2 Configuração técnica
| Campo | Valor |
|---|---|
| Modelo base | unsloth/Qwen2.5-7B-Instruct-bnb-4bit |
| Método | QLoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable % | ~1.05% |
| Hardware | Google Colab Free (Tesla T4 16GB) |
| Tempo total training | <!-- em segundos --> |
| Tamanho final do adapter | <!-- ~80-160 MB --> |
| Localização do adapter | /content/drive/MyDrive/aula06_finetune/adapter_v1/ |

### C.3 Hiperparâmetros
| Campo | Valor |
|---|---|
| Batch size efetivo | 8 (2 × 4 gradient accumulation) |
| Learning rate | 2e-4 |
| Warmup steps | 10 |
| Num epochs | 2 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| LR scheduler | linear |

---

## D) Dataset sintético

### D.1 Origem
| Campo | Valor |
|---|---|
| Seeds usadas | golden_<produto>.csv (50 prompts da Aula 04) |
| Modelo teacher | gpt-4.1-mini (OpenAI) |
| Variações por seed | 10 |
| Total gerado | <!-- bruto --> |
| Pós-filtragem | <!-- limpo --> |
| Taxa de filtragem | <!-- % --> |
| Custo total da síntese | USD <!-- preencher, esperado ~0.70 --> |

### D.2 Filtros aplicados
- Respostas com menos de 40 caracteres descartadas
- Deduplicação por hash MD5 do input
- Exemplos com input ou output vazio rejeitados

### D.3 Distribuição por categoria
<!-- tabela com count por category -->

---

## E) Resultados pós-eval

### E.1 Score ponderado total
| Métrica | Valor |
|---|---|
| Juiz LLM | gpt-4.1 (OpenAI) |
| Score baseline (Aula 04) | <!-- valor --> |
| Score fine-tuned (Aula 06) | <!-- valor --> |
| **Delta** | <!-- +/- valor --> |
| Critério de promoção | delta > +0.2 |
| Promovido? | <!-- SIM / NÃO --> |

### E.2 Delta por dimensão da rubrica

| Dimensão | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| <!-- dim 1 --> | | | |
| <!-- dim 2 --> | | | |
| <!-- dim 3 --> | | | |
| <!-- dim 4 --> | | | |
| <!-- dim 5 --> | | | |

### E.3 Catastrophic forgetting
- Queries que regrediram > 0.3 pontos: <!-- N -->
- Lista das queries com regressão:
  - <!-- E01: -0.45 -->
- Causa provável: <!-- overfitting / lr alto / dataset desbalanceado -->
- Mitigação para próxima iteração: <!-- ex: misturar 15% de dados gerais -->

### E.4 Top 3 queries com maior ganho
| ID | Delta | Insight |
|---|---|---|
| | | |
| | | |
| | | |

---

## F) Decisão e próximos passos

### F.1 Decisão de promoção
**Promover o adapter v1 para uso em produção?** <!-- SIM / NÃO / PARCIAL -->

**Justificativa em 1 frase:**
<!-- preencher -->

### F.2 Trade-off aceito
<!-- O que perdi escolhendo este caminho? Latência? Custo? Generalização? -->

### F.3 O que falsearia esta decisão
<!-- Qual evidência mudaria minha escolha? Ex: blind test humano com colega
contradisendo o juiz, ou re-eval com gpt-5.4 como juiz alternativo. -->

### F.4 Próxima iteração do adapter (v2)
- Dataset: <!-- expandir? mais diverso? misturar dados gerais? -->
- Hyperparams: <!-- mudar r? menos epochs? lr menor? -->
- Método: <!-- testar DPO depois? ORPO? -->

---

## G) Lições da defesa em sala (preencher após Fase 7)
1.
2.

---

## H) Próximas iterações (atualizado)
- [x] Aula 04: avaliação e escolha de modelo
- [x] Aula 05: RAG e defesa contra injection
- [x] Aula 06: fine-tuning QLoRA — esta aula
- [ ] Aula 07: agente carregando o adapter como gerador
- [ ] Aula 08: CI/CD com regression gates sobre o adapter
