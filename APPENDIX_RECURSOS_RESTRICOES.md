# Apendice - Recursos especificos e restricoes

Este apendice acompanha o artefato `ufsc-thesis-sbseg2026` para a avaliacao do CTA/SBSeg 2026. Ele nao contem segredos. Chaves, contas temporarias ou creditos devem ser enviados aos revisores por canal privado da plataforma de submissao, quando necessario.

## Recursos privados necessarios

| Recurso | Necessario para | Como fornecer aos revisores |
| --- | --- | --- |
| `HF_TOKEN` | baixar modelos/datasets gated no Hugging Face | token temporario com permissao minima, se os revisores forem reexecutar downloads |
| `OPENAI_API_KEY` | scoring StrongREJECT e auditoria LLM opcional | chave temporaria com limite de gasto |
| `MARITACA_API_KEY` | geracao com Sabiá-3 | chave temporaria com quota restrita |
| GPU NVIDIA | NLLB/BLASER/vLLM e geracao open-weight | maquina local dos revisores, cluster institucional, ou instancia cloud documentada |

## Restricoes conhecidas

- A reproducao principal nao exige GPU porque parte de `outputs/dataset_frozen.jsonl`.
- A reproducao integral exige aceitar termos de uso de modelos e datasets externos.
- A reproducao integral pode consumir recursos pagos em APIs e GPU.
- Sabiá-3 e API-served; respostas podem variar com a versao do provedor, filtros e data.
- O pacote contem material dual-use de jailbreak. O compartilhamento deve ficar restrito ao contexto de avaliacao.

## Ambiente de referencia usado no experimento

O experimento foi desenhado para dois modos:

- modo analise: Python 3.12 com `uv`, sem GPU;
- modo GPU: Docker baseado em `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, `torch` CUDA e `vllm`.

As configuracoes ficam em:

- `configs/runtime.yaml`;
- `configs/models.yaml`;
- `configs/assets.yaml`;
- `docker/Dockerfile.gpu`.

## Custos e limites

Recomenda-se configurar:

- limites de billing para OpenAI e Maritaca;
- quotas de requisicao conservadoras;
- logs redigidos para evitar vazamento de chaves;
- armazenamento local criptografado caso sejam reexecutadas respostas nocivas.

Arquivos em `usage/` sao agregados/redigidos e existem apenas para contabilidade metodologica. Exports brutos de billing devem permanecer fora do repositorio.

## Procedimento sugerido ao CTA

1. Executar o teste minimo do README.
2. Reproduzir as tabelas a partir de `outputs/dataset_frozen.jsonl`.
3. Conferir uma amostra de linhas em `outputs/translations/`, `outputs/backtranslated/` e `outputs/scored/`.
4. Solicitar recursos privados apenas se for necessario rerodar scoring, Sabiá-3 ou a pipeline completa.

