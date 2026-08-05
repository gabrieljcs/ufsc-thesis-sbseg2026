# Além da Distância Linguística: Jailbreaks Multilíngues em LLMs Especializados por Idioma

Este repositório contém o artefato do artigo *Além da Distância Linguística: Jailbreaks Multilíngues em LLMs Especializados por Idioma*, preparado para o Comitê Técnico de Artefatos do SBSeg 2026. Ele implementa e preserva a avaliação empírica descrita no artigo.

**Resumo do artigo.** Avaliamos se o sucesso de *jailbreaks* multilíngues contra LLMs especializados por idioma é melhor explicado pela distância tipológica ou pela especialização pós-treinamento. O painel contém oito assistentes instrucionais em quatro pares fraco/forte, alinhados a português, italiano, sueco e búlgaro, avaliados em 13 idiomas de ataque. As respostas são pontuadas por StrongREJECT; as análises combinam distância URIEL+, especialização BELEBELE e modelos logísticos de efeitos mistos. Os resultados não sustentam a hipótese direcional de distância: o risco é predominantemente específico do modelo, e os modelos mais fortes do par são, em geral, mais seguros.

# Estrutura do README.md

Este README segue as seções obrigatórias do modelo SBSeg 2026:

- título do projeto e resumo do artigo;
- `Selos Considerados`: selos solicitados e critério de atendimento.
- `Informações básicas`: componentes, ambiente e recursos esperados.
- `Dependências`: software, datasets, modelos, APIs e artefatos de terceiros.
- `Preocupações com segurança`: riscos de conteúdo dual-use, chaves e APIs.
- `Instalação`: preparação do ambiente local e/ou Docker GPU.
- `Teste mínimo`: comandos pequenos para validar a instalação e testar o código.
- `Experimentos`: reivindicações principais e comandos para reproduzi-las.
- `LICENSE`: licença adotada e escopo.

Organização do repositório:

```text
ufsc-thesis-sbseg2026/
├── configs/        modelos, idiomas, perfis de runtime e especificações de assets
├── data/           StrongREJECT importado e matriz URIEL+
├── assets/         datasets locais pequenos; pesos de modelos não são incluídos
├── outputs/        artefatos finais, tabelas, traduções, gerações e dataset congelado
├── docker/         Dockerfile GPU para CUDA/vLLM/NLLB/BLASER
├── scripts/        exportadores LaTeX e scripts auxiliares
├── src/            pacote Python `thesis_eval`
├── tests/          testes unitários e de validação sem inferência pesada
├── usage/          metadados agregados e redigidos de custo/uso
├── APPENDIX_RECURSOS_RESTRICOES.md
├── README.eval-original.md
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
└── uv.lock
```

# Selos Considerados

Os selos considerados são: **Disponíveis (SeloD), Funcionais (SeloF), Sustentáveis (SeloS) e Experimentos Reprodutíveis (SeloR)**.

- **SeloD**: código, configurações, testes, tabelas e principais artefatos congelados estão presentes nesta pasta. Arquivos JSONL grandes também aparecem compactados em `outputs/frozen_jsonl_artifacts.zip` e `outputs/belebele_predictions_jsonl.zip`.
- **SeloF**: o pacote oferece comandos de validação, testes unitários, piloto sem inferência e exportação de tabelas a partir do dataset congelado.
- **SeloS**: a implementação está modularizada em `src/thesis_eval/`, com subpacotes para benchmarks, tradução, geração, avaliação, métricas e análise.
- **SeloR**: as principais reivindicações do artigo podem ser reproduzidas a partir do dataset congelado em tempo viável; a reexecução integral da inferência também é documentada, mas exige GPU, APIs pagas e tempo de processamento substancial.

# Informações básicas

## Objetivo do artefato

O artefato reproduz a pipeline de avaliação usada na tese:

1. importar prompts proibidos do StrongREJECT;
2. traduzir prompts para 13 idiomas com NLLB-200;
3. executar round-trip, BLASER QE e auditoria XSTS;
4. consultar oito modelos-alvo pareados por idioma, mais uma linha de referência em inglês;
5. retrotraduzir respostas para inglês;
6. pontuar respostas com StrongREJECT;
7. calcular URIEL+, BELEBELE IF/CONS/SPEC e diagnósticos de tokenização;
8. congelar o dataset analítico;
9. gerar tabelas e resultados para o capítulo de resultados.

## Modelos avaliados

| ID | Idioma alinhado | Acesso | Identificador |
| --- | --- | --- | --- |
| `sagui_7b` | Português | open weight | `OliveiraJLT/Sagui-7B-Instruct-v0.1` |
| `sabia_3` | Português | API | Maritaca `sabia-3` |
| `llamantino_2_ultrachat_7b` | Italiano | open weight | `swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA` |
| `llamantino_anita_8b` | Italiano | open weight | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` |
| `gpt_sw3` | Sueco | open weight | `AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct` |
| `ai_sweden_llama3_8b` | Sueco | open weight | `AI-Sweden-Models/Llama-3-8B-instruct` |
| `bggpt_7b` | Búlgaro | open weight | `INSAIT-Institute/BgGPT-7B-Instruct-v0.2` |
| `bggpt_gemma_9b` | Búlgaro | open weight | `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0` |
| `llama3_1_8b_reference` | Inglês | open weight | `meta-llama/Llama-3.1-8B-Instruct` |

Idiomas de ataque:

```text
ara bul dan eng fin ita nor por rus spa swa swe ukr
```

## Recursos de hardware

Para testar o código em um computador comum:

- macOS, Linux ou Windows com Python 3.12;
- 2 vCPUs ou mais;
- 4 GB de RAM são suficientes para os testes unitários;
- nenhuma GPU;
- nenhuma chave de API;
- cerca de 2 GB livres para o checkout com os artefatos versionados.

Para reproduzir as análises a partir do dataset congelado:

- CPU comum, sem GPU;
- 8 GB de RAM recomendados;
- `uv sync --extra analysis`;
- os JSONL expandidos podem ser restaurados com `unzip` quando necessário.

Para reexecutar tradução, BLASER, vLLM ou geração open-weight:

- Linux com Docker e NVIDIA Container Toolkit;
- GPU NVIDIA com CUDA; A100 40 GB ou superior foi o perfil usado no experimento original;
- 150-250 GB livres se todos os pesos forem baixados localmente;
- tempo de execução de horas a dias para a pipeline completa, dependendo de filas de API e GPU.

## Dataset congelado incluído

- `outputs/dataset_frozen.parquet`;
- `outputs/frozen_jsonl_artifacts.zip`;
- `outputs/belebele_predictions_jsonl.zip`;
- `outputs/tables/*.csv`;
- traduções, auditorias, retrotraduções, gerações e pontuações por modelo/idioma.

No repositório GitHub, `outputs/dataset_frozen.jsonl` e `outputs/scored/all_strongreject_scores.jsonl` devem ser restaurados com:

```sh
unzip -n outputs/frozen_jsonl_artifacts.zip
```

Esses JSONL expandidos excedem o limite individual de arquivo do GitHub, mas o ZIP correspondente está versionado.

# Dependências

## Ferramentas de sistema

- Python `>=3.12,<3.13`;
- `uv` para instalação reprodutível;
- `git` e `unzip`;
- Docker, apenas para etapas GPU;
- NVIDIA driver, CUDA e NVIDIA Container Toolkit, apenas para etapas GPU.

## Dependências Python principais

As dependências estão fixadas em `pyproject.toml` e `uv.lock`. Os grupos relevantes são:

- base: `pyyaml`, `requests`, `datasets`, `huggingface_hub`, `numpy`, `pandas`;
- `analysis`: `statsmodels`, `scipy`, `pyarrow`;
- `scoring`: pacote oficial `strong-reject`, necessário somente para reexecutar o scoring StrongREJECT;
- `mac`: `torch`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`;
- `rtx`: `torch`, `vllm`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`;
- `qc`: `torch`, `fairseq2`, `sonar-space`.

## Benchmarks, modelos e serviços externos

- StrongREJECT: fonte primária de prompts nocivos e protocolo de scoring.
- NLLB-200: tradução de prompts e retrotradução de respostas.
- BLASER 2.0 QE: estimativa de qualidade de tradução.
- BELEBELE: proxy benigno IF/CONS/SPEC.
- URIEL+: distância tipológica.
- Hugging Face: download dos modelos open-weight e datasets.
- OpenAI API: juiz StrongREJECT e auditoria LLM opcional.
- Maritaca API: acesso ao Sabiá-3.

As credenciais devem ser definidas em `.env` ou no ambiente somente quando uma etapa externa for executada:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

# Preocupações com segurança

Este artefato contém prompts nocivos, respostas de modelos a tentativas de jailbreak e pontuações de conformidade insegura.

Cuidados recomendados:

- não publicar trechos de prompts/respostas fora do contexto de avaliação;
- não executar a geração contra APIs de produção sem limites de custo e monitoramento;
- não reutilizar as respostas como instruções operacionais;
- tratar provider blocks e falhas técnicas como metadados, não como recusas do modelo.

O dataset congelado já inclui resultados suficientes para reproduzir as análises principais sem consultar modelos novamente. Esse é o caminho recomendado para a revisão inicial.

# Instalação

Os comandos abaixo assumem que o terminal está na raiz do repositório que contém esta pasta.

```sh
cd ufsc-thesis-sbseg2026
```

## Instalação leve para testar o código

Esta é a instalação recomendada para computadores básicos. Ela não baixa pesos de modelos, não usa GPU e não chama APIs.

```sh
uv sync
```

Esse ambiente é suficiente para:

- validar configurações;
- executar os testes unitários;
- rodar a pipeline piloto com dados simulados;
- inspecionar os artefatos congelados.

## Instalação para análise estatística sem GPU

Use esta instalação para exportar tabelas e ajustar modelos estatísticos a partir do dataset congelado.

```sh
uv sync --extra analysis
```

## Instalação para scoring ou auditoria remota

Use o extra `scoring` somente se for reexecutar o avaliador StrongREJECT ou etapas que dependem do pacote oficial de scoring.

```sh
uv sync --extra analysis --extra scoring
```

Configure credenciais somente se for executar scoring, auditoria externa ou API:

```sh
cp .env.example .env
# editar .env localmente, sem versionar
```

## Instalação para smoke test local em Apple Silicon

```sh
uv sync --extra mac --extra qc --extra analysis
```

## Imagem Docker GPU recomendada

Use esta imagem para NLLB, BLASER e geração vLLM:

```sh
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
```

O comando `thesis-eval-gpu` encapsula `docker run` e monta o diretório atual para preservar `assets/`, `data/` e `outputs/` entre execuções.

# Teste mínimo

Este teste valida o código em CPU e não baixa modelos grandes, não consulta APIs e não exige GPU.

Tempo esperado: menos de 1 minuto após `uv sync` em um notebook comum. Na preparação deste artefato, `uv run python -m unittest` executou 91 testes, com 9 skips esperados para caminhos opcionais.

Recursos esperados: CPU comum e cerca de 2 GB de RAM durante os testes.

```sh
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run python -m unittest
uv run thesis-eval run-pilot --output-dir outputs/pilot
```

Resultado esperado:

- `validate-config` deve indicar `OK: 8 paired models, 1 reference baseline, 13 languages`;
- os testes unitários devem terminar sem falhas;
- `run-pilot` deve criar artefatos pequenos em `outputs/pilot/`;
- nenhuma chamada externa de modelo é necessária para esse teste.

# Experimentos

## Reivindicação #1 - O dataset congelado está íntegro e gera as tabelas principais

Objetivo: reproduzir as tabelas descritivas e de cobertura usadas no capítulo de resultados, sem reexecutar inferência.

Recursos: CPU, 8 GB RAM, sem GPU. Tempo esperado: 5-20 minutos.

Comandos:

```sh
uv sync --extra analysis

unzip -n outputs/frozen_jsonl_artifacts.zip
unzip -n outputs/belebele_predictions_jsonl.zip

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables

uv run thesis-eval export-results \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/dataset_frozen.parquet \
  --allow-frozen-overwrite
```

Resultado esperado:

- `outputs/tables/results_coverage.csv`;
- `outputs/tables/asr_by_model_language.csv`;
- `outputs/tables/crosslingual_asr_by_model_language.csv`;
- `outputs/tables/translation_qc_arithmetic.csv`;
- `outputs/dataset_frozen.parquet`.

## Reivindicação #2 - ASR, distância URIEL+ e BELEBELE SPEC reproduzem os insumos de análise

Objetivo: gerar novamente os arquivos de correlação e diagnósticos que sustentam as figuras e tabelas de ASR, distância, SPEC e tokenização.

Recursos: CPU, 8 GB RAM, sem GPU. Tempo esperado: 5-20 minutos.

Comandos:

```sh
uv sync --extra analysis

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables

uv run python scripts/write_uriel_latex_assets.py
uv run python scripts/write_thesis_results_latex_assets.py
```

Resultado esperado:

- `outputs/tables/distance_asr_correlation.csv`;
- `outputs/tables/spec_asr_correlation.csv`;
- `outputs/tables/tokenizer_diagnostics.csv`;
- `outputs/tables/closest_farthest_languages.csv`;
- atualização dos assets LaTeX consumidos pela tese.

## Reivindicação #3 - Os modelos estatísticos e as sensibilidades pós-revisão podem ser ajustados a partir do painel congelado

Objetivo: reproduzir os efeitos principais, diagnósticos de convergência e colinearidade, estratos weak/strong, robustez de tokenização, decomposição IF/CONS e sensibilidades solicitadas na revisão.

Recursos: CPU, 8-16 GB RAM, sem GPU. Tempo esperado: 10-60 minutos, dependendo da máquina.

Comandos:

```sh
uv sync --extra analysis

uv run thesis-eval fit-glmm-suite \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables
```

Resultado esperado:

- `outputs/tables/glmm_main_effects.csv`;
- `outputs/tables/glmm_diagnostics.csv`;
- `outputs/tables/glmm_gee_sensitivity.csv`;
- `outputs/tables/glmm_prior_sensitivity.csv`;
- `outputs/tables/glmm_postreview_sensitivity.csv`;
- `outputs/tables/glmm_aggregated_cell_sensitivity.csv`;
- `outputs/tables/glmm_collinearity.csv`;
- `outputs/tables/glmm_strata_effects.csv`;
- `outputs/tables/glmm_spec_components.csv`;
- `outputs/tables/glmm_tokenizer_robustness.csv`;
- `outputs/tables/prereg_distance_slope_retention.csv`;
- `outputs/tables/prereg_falsification_summary.csv`.

O ajuste principal usa Laplace/MAP com múltiplos inícios BFGS e é aceito apenas se a otimização tiver sucesso, a norma do gradiente for no máximo `1e-4`, os diagnósticos forem finitos e a covariância de Laplace for positiva definida. O mesmo comando também exporta a GEE agrupada por *prompt*, uma sensibilidade de prior e sensibilidades que removem a célula alinhada, os idiomas de maior risco de tradução e todas as linhas sinalizadas por BLASER. A análise agregada em 103 células modelo--idioma verifica se os sinais se preservam sem tratar repetições por *prompt* como novas variações dos preditores de célula.

## Reivindicação #4 - A pipeline completa pode ser reexecutada, com recursos externos

Objetivo: rerodar tradução, auditoria, geração, retrotradução, scoring, BELEBELE, congelamento e análise.

Recursos: GPU NVIDIA, Docker, Hugging Face, OpenAI API, Maritaca API, espaço em disco para pesos de modelos e tempo substancial. Este experimento não é recomendado como primeiro teste do CTA.

Sequência de alto nível:

```sh
uv run thesis-eval validate-config
uv run thesis-eval import-strongreject --source github --pilot-size 2
uv run thesis-eval download-assets --name belebele --name multijail
uv run thesis-eval verify-assets --name strongreject --name belebele --name multijail

uv run thesis-eval-gpu download-assets --name nllb_200_3_3b --name blaser_2_0_qe
uv run thesis-eval-gpu prewarm-sonar
uv run thesis-eval prepare-uriel --from-csv
```

Depois execute as etapas documentadas em `README.eval-original.md`, na ordem:

1. `translate`;
2. `prompt-roundtrip`;
3. `translation-qc`;
4. `export-audit-queue`, auditoria humana ou `judge-audit-queue`, `import-audit`;
5. `generate-targets`;
6. `backtranslate-responses`;
7. `score-strongreject` ou caminho OpenAI Batch;
8. `predict-belebele`, `repair-belebele-predictions`, `compute-spec`;
9. `build-dataset`;
10. `export-report-tables` e `fit-glmm`.

Resultado esperado: recriação dos mesmos tipos de artefatos sob `outputs/`, com pequenas variações possíveis em respostas API e scoring remoto por causa de versões de provedores e datas de execução.

# LICENSE

Este repositório adota a licença MIT para o código-fonte e a documentação autoral do pacote. A licença completa está em `LICENSE`.

Dados, modelos, benchmarks e saídas derivadas de componentes de terceiros não são re-licenciados por este repositório. Consulte `THIRD_PARTY_NOTICES.md` e os termos de origem de StrongREJECT, BELEBELE, NLLB-200, BLASER 2.0 QE, URIEL+, modelos Hugging Face, OpenAI API e Maritaca API.
