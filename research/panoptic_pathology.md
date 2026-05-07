# 🔬 Panoptic Segmentation of nUclei and tissue in advanced MelanomA (PUMA)

## 📖 1. Contexto Clínico e Tese (Background)

O Melanoma é um dos cânceres de pele mais agressivos. O sucesso do tratamento, especialmente a imunoterapia em casos avançados, depende da resposta imunológica do hospedeiro. 

> [!IMPORTANT]
> **A Tese do PUMA:** A presença e, crucialmente, a **localização espacial** de Linfócitos Infiltrantes de Tumor (TILs) são biomarcadores preditivos de sobrevivência. Métodos manuais de score são subjetivos; a IA surge para trazer objetividade e granularidade (intratumoral vs. estromal).

---

## 📊 2. Especificações do Dataset

O dataset PUMA é composto por uma infraestrutura de dados multi-escala:

| Atributo | Detalhe Técnico |
| :--- | :--- |
| **Volume** | 310 ROIs (155 Primários / 155 Metastáticos) |
| **Resolução Principal** | 1024 x 1024 pixels (40x magnification) |
| **Contexto Global** | 5120 x 5120 pixels (Centrados na ROI) |
| **Labels** | GeoJSON (Polígonos de alta precisão) |
| **Validação** | Dermatopatologista sênior (Board-certified) |

---

## 🛠️ 3. Definição das Tarefas (Tracks)

O desafio exige a resolução simultânea de dois problemas de Visão Computacional de naturezas distintas:

### 🟢 Task 1: Semantic Tissue Segmentation (Macro)
Segmentação de regiões densas de tecido.
- **Classes:** Estroma, Vasos Sanguíneos, Tumor, Epitélio (Epiderme) e Necrose.
- **Background:** Classe 0 (Ignorada na métrica).

### 🔵 Task 2: Nuclei Instance Segmentation (Micro)
Detecção e segmentação individual de núcleos celulares.

| Nível | Classes de Instância |
| :--- | :--- |
| **Track 1** | Tumor, TILs (Linfócitos/Plasma), Outros |
| **Track 2** | 10 classes detalhadas (Neutrófilos, Endotélio, Apoptóticos, etc.) |

---

## 📈 4. Protocolo de Avaliação (Métricas de Sucesso)

O Arquiteto deve otimizar o modelo para duas métricas conflitantes:

### 📐 Macro-F1 Score (Task 2)
Baseado em um critério de **"Hit" Geométrico**:
1. **Extração de Centroides:** Polígonos são convertidos em pontos (x, y).
2. **Raio de Censura (15px):** Uma predição só é válida se o seu centroide estiver a menos de 15 pixels de um Ground Truth.
3. **Matching:** Prioridade por `confidence_score` e proximidade.
4. **Alinhamento de Classe:** Se o match ocorrer mas a classe divergir, é contabilizado como **False Positive (FP)**.

### 🎲 Micro-Dice Score (Task 1)
Cálculo de interseção sobre união (IoU) para as classes de tecido, concatenando todos os resultados de segmentação antes do cálculo da média.

---

## 🏗️ 5. Implicações para a Arquitetura (Architectural Insights)

> [!TIP]
> **Estratégia DinoV2:** Como o backbone será *frozen*, a qualidade da segmentação panóptica dependerá inteiramente da capacidade do "Head" em decodificar as features do ViT para duas escalas:
> 1. **Local (Nuclei):** Requer alta resolução espacial (Patch size 14 é ideal).
> 2. **Global (Tissue):** Requer campos receptivos amplos para entender a arquitetura do estroma vs tumor.

### Desafios de Engenharia Identificados:
- **Conversão GeoJSON → Mask:** Necessidade de rasterização eficiente para o treino.
- **Duality Loss:** Equilibrar uma loss de pixel-wise cross-entropy (Tecido) com uma loss de detecção/segmentação de instâncias (Núcleos).
- **Gestão de Memória:** O pipeline deve suportar o volume de 5120x5120 (Context) em setups de 16GB RAM.