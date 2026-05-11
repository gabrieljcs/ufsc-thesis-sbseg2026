# Artefato SBSeg 2026 - Avaliacao Cross-Lingual Jailbreak

Este repositorio contem o pacote de avaliacao reprodutivel da tese *Multilingual Jailbreaks in Language-Specific Large Language Models*, preparado como artefato para o Comitê Tecnico de Artefatos do SBSeg 2026.

O artefato implementa e disponibiliza a pipeline empirica usada para avaliar se o sucesso de jailbreaks cross-lingual contra LLMs alinhados para idiomas especificos e melhor explicado por distancia linguistica, especializacao pos-treinamento, fragmentacao de tokenizacao e instabilidade de traducao/pontuacao. A copia foi criada a partir de `eval/`; o diretório original não deve ser usado pelos avaliadores desta submissao.

Fonte das regras de estruturacao deste README: introducao do CTA/SBSeg 2026 em <https://doc-artefatos.github.io/sbseg2026/introducao.html> e instrucoes de submissao em <https://doc-artefatos.github.io/sbseg2026/subinstrucoes.html>.

# Estrutura do readme.md

Este README segue as secoes obrigatorias do modelo SBSeg 2026:

- `Selos Considerados`: selos solicitados e criterio de atendimento.
- `Informacoes basicas`: componentes, ambiente e recursos esperados.
- `Dependencias`: software, datasets, modelos, APIs e artefatos de terceiros.
- `Preocupacoes com seguranca`: riscos de conteudo dual-use, chaves e APIs.
- `Instalacao`: preparacao do ambiente local e/ou Docker GPU.
- `Teste minimo`: comandos pequenos para validar a instalacao.
- `Experimentos`: reivindicacoes principais e comandos para reproduzi-las.
- `LICENSE`: estado de licenciamento do pacote e componentes de terceiros.

Organizacao do repositorio:

```text
ufsc-thesis-sbseg2026/
├── configs/        modelos, idiomas, perfis de runtime e especificacoes de assets
├── data/           StrongREJECT importado e matriz URIEL+
├── assets/         datasets locais pequenos; pesos de modelos nao sao incluidos
├── outputs/        artefatos finais, tabelas, traducoes, geracoes e dataset congelado
├── docker/         Dockerfile GPU para CUDA/vLLM/NLLB/BLASER
├── scripts/        exportadores LaTeX e scripts auxiliares
├── src/            pacote Python `thesis_eval`
├── tests/          testes unitarios e de validacao sem inferencia pesada
├── usage/          metadados agregados e redigidos de custo/uso
├── APPENDIX_RECURSOS_RESTRICOES.md
├── README.eval-original.md
├── pyproject.toml
└── uv.lock
```

`README.eval-original.md` preserva a documentacao operacional original da pasta `eval/`, com a sequencia longa de comandos por etapa. Este README e a entrada principal para avaliadores do SBSeg.

# Selos Considerados

Os selos considerados sao: **Disponiveis (SeloD), Funcionais (SeloF), Sustentaveis (SeloS) e Experimentos Reprodutiveis (SeloR)**.

- **SeloD**: codigo, configuracoes, testes, tabelas e principais artefatos congelados estao presentes nesta pasta. Arquivos JSONL grandes tambem aparecem compactados em `outputs/frozen_jsonl_artifacts.zip` e `outputs/belebele_predictions_jsonl.zip`.
- **SeloF**: o pacote oferece comandos de validacao, testes unitarios, piloto sem inferencia e exportacao de tabelas a partir do dataset congelado.
- **SeloS**: a implementacao esta modularizada em `src/thesis_eval/`, com subpacotes para benchmarks, traducao, geracao, avaliacao, metricas e analise.
- **SeloR**: as principais reivindicacoes do artigo podem ser reproduzidas a partir do dataset congelado em tempo viavel; a reexecucao integral da inferencia tambem e documentada, mas exige GPU, APIs e tempo de processamento substancial.

# Informações básicas

## Objetivo do artefato

O artefato reproduz a pipeline de avaliacao usada na tese:

1. importar prompts proibidos do StrongREJECT;
2. traduzir prompts para 13 idiomas com NLLB-200;
3. executar round-trip, BLASER QE e auditoria XSTS;
4. consultar oito modelos-alvo pareados por idioma, mais uma linha de referencia em ingles;
5. retrotraduzir respostas para ingles;
6. pontuar respostas com StrongREJECT;
7. calcular URIEL+, BELEBELE IF/CONS/SPEC e diagnosticos de tokenizacao;
8. congelar o dataset analitico;
9. gerar tabelas e resultados para o capitulo de resultados.

## Modelos avaliados

| ID | Idioma alinhado | Acesso | Identificador |
| --- | --- | --- | --- |
| `sagui_7b` | Portugues | open weight | `OliveiraJLT/Sagui-7B-Instruct-v0.1` |
| `sabia_3` | Portugues | API | Maritaca `sabia-3` |
| `llamantino_2_ultrachat_7b` | Italiano | open weight | `swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA` |
| `llamantino_anita_8b` | Italiano | open weight | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` |
| `gpt_sw3` | Sueco | open weight | `AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct` |
| `ai_sweden_llama3_8b` | Sueco | open weight | `AI-Sweden-Models/Llama-3-8B-instruct` |
| `bggpt_7b` | Bulgaro | open weight | `INSAIT-Institute/BgGPT-7B-Instruct-v0.2` |
| `bggpt_gemma_9b` | Bulgaro | open weight | `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0` |
| `llama3_1_8b_reference` | Ingles | open weight | `meta-llama/Llama-3.1-8B-Instruct` |

Idiomas de ataque:

```text
ara bul dan eng fin ita nor por rus spa swa swe ukr
```

## Recursos de hardware

Para avaliacao rapida a partir dos resultados congelados:

- macOS, Linux ou Windows com Python 3.12;
- 2 vCPUs ou mais;
- 8 GB de RAM recomendados;
- cerca de 2 GB livres para este pacote ja copiado;
- nenhuma GPU.

Para reexecutar traducao, BLASER, vLLM ou geracao open-weight:

- Linux com Docker e NVIDIA Container Toolkit;
- GPU NVIDIA com CUDA; A100 40 GB ou superior e o perfil usado no experimento original;
- 150-250 GB livres se todos os pesos forem baixados localmente;
- tempo de execucao de horas a dias para a pipeline completa, dependendo de filas de API e GPU.

## Conteudo congelado incluido

O pacote copiado contem, entre outros:

- `outputs/dataset_frozen.parquet`;
- `outputs/frozen_jsonl_artifacts.zip`;
- `outputs/belebele_predictions_jsonl.zip`;
- `outputs/tables/*.csv`;
- traducoes, auditorias, retrotraducoes, geracoes e pontuacoes por modelo/idioma.

No repositorio GitHub, `outputs/dataset_frozen.jsonl` e
`outputs/scored/all_strongreject_scores.jsonl` devem ser restaurados com
`unzip -n outputs/frozen_jsonl_artifacts.zip`, pois os JSONL expandidos excedem
o limite individual de arquivo do GitHub.

# Dependências

## Ferramentas de sistema

- Python `>=3.12,<3.13`;
- `uv` para instalacao reprodutivel;
- `git` e `unzip`;
- Docker, para etapas GPU;
- NVIDIA driver, CUDA e NVIDIA Container Toolkit, apenas para etapas GPU.

## Dependencias Python principais

As dependencias estao fixadas em `pyproject.toml` e `uv.lock`. Os grupos relevantes sao:

- base: `pyyaml`, `requests`, `datasets`, `huggingface_hub`, `numpy`, `pandas`;
- `analysis`: `statsmodels`, `scipy`, `pyarrow`;
- `scoring`: pacote oficial `strong-reject`;
- `mac`: `torch`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`;
- `rtx`: `torch`, `vllm`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`;
- `qc`: `torch`, `fairseq2`, `sonar-space`.

## Benchmarks, modelos e servicos externos

- StrongREJECT: fonte primaria de prompts nocivos e protocolo de scoring.
- NLLB-200: traducao de prompts e retrotraducao de respostas.
- BLASER 2.0 QE: estimativa de qualidade de traducao.
- BELEBELE: proxy benigno IF/CONS/SPEC.
- URIEL+: distancia tipologica.
- Hugging Face: download dos modelos open-weight e datasets.
- OpenAI API: juiz StrongREJECT e auditoria LLM opcional.
- Maritaca API: acesso ao Sabiá-3.

As credenciais devem ser definidas em `.env` ou no ambiente:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

Nenhuma chave real deve ser versionada ou enviada aos revisores dentro do repositorio publico. Quando o CTA precisar de acesso privado, use o apendice conforme indicado em `APPENDIX_RECURSOS_RESTRICOES.md`.

# Preocupações com segurança

Este artefato contem prompts nocivos, respostas de modelos a tentativas de jailbreak e pontuacoes de conformidade insegura. Esses dados sao necessarios para auditar a metodologia, mas sao material dual-use.

Cuidados recomendados:

- nao publicar trechos de prompts/respostas fora do contexto de avaliacao;
- nao executar a geracao contra APIs de producao sem limites de custo e monitoramento;
- nao reutilizar as respostas como instrucoes operacionais;
- manter `.env`, chaves de API, tokens Hugging Face e exports brutos de billing fora do repositorio;
- executar a pipeline completa em ambiente isolado, preferencialmente VM ou container;
- tratar provider blocks e falhas tecnicas como metadados, nao como recusas do modelo.

O dataset congelado ja inclui resultados suficientes para reproduzir as analises principais sem consultar modelos novamente. Esse e o caminho recomendado para a revisao inicial.

# Instalação

Os comandos abaixo assumem que o terminal esta na raiz do repositorio que contem esta pasta.

```sh
cd ufsc-thesis-sbseg2026
```

## Instalacao para analise sem GPU

Use esta instalacao para validar configuracoes, rodar testes, exportar tabelas e ajustar modelos estatisticos a partir do dataset congelado.

```sh
uv sync --extra analysis --extra scoring
```

Configure credenciais somente se for executar scoring/auditoria/API:

```sh
cp .env.example .env
# editar .env localmente, sem versionar
```

## Instalacao para smoke test local em Apple Silicon

```sh
uv sync --extra mac --extra qc --extra scoring --extra analysis
```

## Imagem Docker GPU recomendada

Use esta imagem para NLLB, BLASER e geracao vLLM:

```sh
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
```

O comando `thesis-eval-gpu` encapsula `docker run` e monta o diretorio atual para preservar `assets/`, `data/` e `outputs/` entre execucoes.

# Teste mínimo

Este teste nao baixa modelos grandes nem consulta APIs. Ele valida a instalacao, configuracoes e a pipeline mockada.

Tempo esperado: 2-10 minutos em notebook comum, apos `uv sync`.

Recursos esperados: ate 2 GB de RAM, sem GPU.

```sh
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval list-assets
uv run python -m unittest
uv run thesis-eval run-pilot --output-dir outputs/pilot
```

Resultado esperado:

- `validate-config` deve indicar `OK: 8 paired models, 1 reference baseline, 13 languages`;
- os testes unitarios devem terminar sem falhas;
- `run-pilot` deve criar artefatos pequenos em `outputs/pilot/`;
- nenhuma chamada externa de modelo e necessaria para esse teste.

# Experimentos

## Reivindicação #1 - O dataset congelado esta integro e gera as tabelas principais

Objetivo: reproduzir as tabelas descritivas e de cobertura usadas no capitulo de resultados, sem reexecutar inferencia.

Recursos: CPU, 8 GB RAM, sem GPU. Tempo esperado: 5-20 minutos.

Comandos:

```sh
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

## Reivindicação #2 - ASR, distancia URIEL+ e BELEBELE SPEC reproduzem os insumos de analise

Objetivo: gerar novamente os arquivos de correlacao e diagnosticos que sustentam as figuras e tabelas de ASR, distancia, SPEC e tokenizacao.

Recursos: CPU, 8 GB RAM, sem GPU. Tempo esperado: 5-20 minutos.

Comandos:

```sh
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
- atualizacao dos assets LaTeX consumidos pela tese.

## Reivindicação #3 - Os modelos estatisticos principais podem ser ajustados a partir do painel congelado

Objetivo: reproduzir os efeitos principais, diagnosticos de colinearidade, estratos weak/strong, robustez de tokenizacao e decomposicao IF/CONS.

Recursos: CPU, 8-16 GB RAM, sem GPU. Tempo esperado: 10-60 minutos, dependendo da maquina.

Comandos:

```sh
uv run thesis-eval fit-glmm \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/tables/glmm_main_effects.csv

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables
```

Resultado esperado:

- `outputs/tables/glmm_main_effects.csv`;
- `outputs/tables/glmm_collinearity.csv`;
- `outputs/tables/glmm_strata_effects.csv`;
- `outputs/tables/glmm_spec_components.csv`;
- `outputs/tables/glmm_tokenizer_robustness.csv`;
- `outputs/tables/prereg_distance_slope_retention.csv`;
- `outputs/tables/prereg_falsification_summary.csv`.

## Reivindicação #4 - A pipeline completa pode ser reexecutada, com recursos externos

Objetivo: rerodar traducao, auditoria, geracao, retrotraducao, scoring, BELEBELE, congelamento e analise.

Recursos: GPU NVIDIA, Docker, Hugging Face, OpenAI API, Maritaca API, disco para pesos de modelos e tempo substancial. Este experimento nao e recomendado como primeiro teste do CTA.

Sequencia de alto nivel:

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

Resultado esperado: recriacao dos mesmos tipos de artefatos sob `outputs/`, com pequenas variacoes possiveis em respostas API e scoring remoto por causa de versoes de provedores e datas de execucao.

# LICENSE

O diretorio original `eval/` nao declarava uma licenca formal. Para esta copia SBSeg, o arquivo `LICENSE` registra uma posicao conservadora: todos os direitos reservados pelos autores ate que uma licenca publica definitiva seja escolhida.

Componentes de terceiros continuam sujeitos as respectivas licencas e termos de uso, incluindo StrongREJECT, BELEBELE, NLLB-200, BLASER 2.0 QE, URIEL+, modelos Hugging Face, OpenAI API e Maritaca API.
