# CP2 · Escala Computacional em Redes Neurais

**IA e Machine Learning — Engenharia de Software** · ANN em scikit-learn e PyTorch · CPU × GPU · custo × desempenho

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucksza/CP5-ML/blob/main/CP2_Escala_Computacional_Redes_Neurais.ipynb)

## Integrantes

| # | Integrante | RM |
|---|---|---|
| 1 | Lucca Borges | 554608 |
| 2 | Ruan Melo | 557599 |
| 3 | Rodrigo Jimenez | 558148 |
| 4 | João Victor Franco | 556790 |
| 5 | Bruno Leão | 555563 |

## Entrega

- **Documento:** [`CP2_Escala_Computacional_Redes_Neurais.pdf`](CP2_Escala_Computacional_Redes_Neurais.pdf) — enunciado preenchido com as Tabelas A e B (valores reais), os gráficos, as respostas
  às questões 1 a 8, a conclusão e, no anexo, o protocolo e a cópia da saída da execução.
- **Notebook executado, com todas as saídas:** [`CP2_Escala_Computacional_Redes_Neurais.ipynb`](CP2_Escala_Computacional_Redes_Neurais.ipynb)
  — Etapas A, B e C, tabelas, gráficos, evidências numéricas, análise complementar e checagem das respostas.
- **Dados brutos:** [`resultados/`](resultados/) (CSVs com tempos mín./máx. de cada medição, `ambiente.json`,
  `selecao_features.json` e as figuras em `resultados/figuras/`). Dependências em [`requirements.txt`](requirements.txt).

## Resultado em resumo

- Crescer a rede de 33 para 132.609 parâmetros
  **não trouxe ganho consistente de acurácia**: o maior ganho pontual da (256,256,256) sobre a (8,) (+2,63 p.p.)
  desaparece na média de 10 sorteios, e a melhor relação custo × benefício ficou com redes pequenas.
- O que melhorou o resultado de forma consistente foi passar de **2 para 22 features** (+4,39 a +10,53 p.p.).
- Neste problema pequeno, a **GPU foi mais lenta que a CPU nas 8 configurações** (speedup de 0,40 a
  0,80); ela só passou a compensar quando cada passo de treino teve muito mais cálculo
  (até 12,1× com camadas de 2048 neurônios).

## Como executar

- **Google Colab:** botão *Open in Colab* acima → *Ambiente de execução* → *Alterar tipo de ambiente de execução* →
  *Acelerador de hardware* → **GPU** → *Executar tudo*. O dataset vem do próprio scikit-learn (não há upload). Sem GPU, o
  notebook registra a indisponibilidade e segue com o que dá para medir.
- **Local:** `pip install -r requirements.txt` (para GPU NVIDIA, instale o `torch` com CUDA pelo índice oficial) e
  `jupyter nbconvert --to notebook --execute --inplace CP2_Escala_Computacional_Redes_Neurais.ipynb`.
- A execução registrada levou cerca de 5 minutos; em outro hardware o tempo muda. Uma nova execução recalcula as seções 1–10;
  as respostas (seção 11) foram redigidas sobre a execução registrada, e a **célula de checagem** da seção 11 mostra se a
  interpretação continua válida e se os números citados ainda são os da execução corrente (✔/✘).

## Ambiente da execução registrada

| Item | Valor |
|---|---|
| CPU | 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz — 8 núcleos físicos / 16 lógicos (PyTorch usando 8 threads) |
| GPU | NVIDIA GeForce RTX 3060 · 12,0 GB (CUDA 13.0) |
| RAM | 31,9 GB |
| Sistema | Windows 11 (build 26200) · Python 3.11.9 |
| Bibliotecas | numpy 2.4.6 · pandas 3.0.6 · scikit-learn 1.9.1 · torch 2.14.1+cu130 |
| Data | 2026-10-14 08:16 |

## Protocolo

- Dados: Breast Cancer Wisconsin (Diagnostic), 569 exames, `diagnosis` 1 = maligno / 0 = benigno.
- `dados_reduzidos` (22 features): o enunciado usa um `dados_reduzidos` com 22 features; aqui ele é construído por um filtro de redundância — de cada par com |correlação| > 0,92 sai a segunda coluna —, calculado só nas 455 linhas de treino do sorteio principal. O limiar foi escolhido por levar exatamente às 22 features; a equivalência com a lista do notebook da disciplina não foi verificada. Removidas: `perimeter_mean`, `area_mean`, `concave points_mean`, `perimeter_se`, `area_se`, `radius_worst`, `perimeter_worst`, `area_worst`.
- Etapas: A com `radius_mean` e `texture_mean`; B com `X = dados_reduzidos.drop(columns=['diagnosis'])`; C com a mesma ANN do PyTorch em CPU e GPU.
- Separação: 80% treino / 20% teste (114 exames), estratificada, `random_state=42`; `StandardScaler` ajustado só no treino.
- Redes: (8,), (64,64), (128,128,128), (256,256,256) com ReLU e 1 saída; Adam (lr = 0,001) e lotes de 200 nos dois frameworks. O scikit-learn treina até atingir o critério de parada (teto de 1000 épocas); o PyTorch treina 200 épocas fixas com `BCEWithLogitsLoss`.
- Tempo: só o treino, mediana de 5 execuções após 1 aquecimento; na GPU, com `torch.cuda.synchronize()` e incluindo a cópia de modelo e dados para a placa. Experimento B: mesma rede, mesmos dados, épocas, perda, otimizador, pesos iniciais e ordem dos lotes — só o dispositivo muda.
