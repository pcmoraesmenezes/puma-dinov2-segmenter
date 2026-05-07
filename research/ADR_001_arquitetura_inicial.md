# ADR 001: Escolha da Arquitetura de Decodificação (PUMA)

**Status:** Decidido  
**Data:** 2026-05-06  
**Responsável:** Paulo Menezes (Arquiteto de IA)

## 1. Contexto do Problema
O desafio PUMA exige a resolução de duas tarefas simultâneas:
1. **Task 1 (Macro):** Segmentação semântica de tecidos (Tumor, Estroma, etc).
2. **Task 2 (Micro):** Segmentação/Detecção de instâncias de núcleos celulares.

O hardware disponível é limitado a **16GB de RAM**, o que impõe uma restrição severa ao tamanho do modelo durante o treinamento.

## 2. Dicionário de Termos
Para garantir clareza absoluta, definimos as peças da nossa arquitetura:

*   **Backbone (DINOv2):** O "Cérebro" pré-treinado que enxerga a imagem. Ele permanecerá **Frozen** (congelado) para economizar memória e alavancar o conhecimento prévio do Foundation Model.
*   **Head (Cabeça):** A parte final do modelo que toma a decisão e desenha o resultado.
*   **Linear:** Uma única camada matemática de saída. Rápida, mas limitada em detalhes.
*   **Hybrid (Híbrida):** Uma pequena sequência de camadas (3 a 4) que combina informações de diferentes níveis do Backbone.
*   **Mask2Former:** Uma arquitetura de decodificação massiva e complexa, padrão SOTA atual.

## 3. Comparativo Técnico

| Atributo | Linear Heads | Mask2Former | Hybrid Head (Escolha) |
| :--- | :--- | :--- | :--- |
| **Complexidade de Código** | Baixa | Altíssima | Média |
| **Consumo de Memória (VRAM)** | Mínimo (< 4GB) | Máximo (> 12GB) | Moderado (~6GB) |
| **Resolução de Detalhes** | Baixa (Míope) | Altíssima | Alta (Focada) |
| **Viabilidade (16GB RAM)** | Total | Baixa/Nula | Alta |

## 4. Decisão: Arquitetura Híbrida (Hybrid Head)
A decisão é implementar uma **Hybrid Head** conectada ao backbone **DINOv2 (ViT-S/14)**.

**Racional:**
1. **Precisão Micro:** A segmentação de núcleos no PUMA exige precisão de centroide (limite de 15px). Uma cabeça Linear pura dificilmente atingiria a resolução necessária.
2. **Eficiência de Recurso:** A arquitetura híbrida permite extrair features de múltiplos blocos do DINOv2 (multiscale) sem a sobrecarga computacional de um Transformer Decoder completo como o Mask2Former.
3. **Escalabilidade:** Permite evoluir para um pipeline de *Feature Caching* caso o treinamento direto ainda se prove pesado.

## 5. Implicações
* **Engenharia de Dados:** O pipeline deve preparar máscaras de tecido e mapas de calor (heatmaps) para os núcleos.
* **Treinamento:** Otimizaremos apenas os pesos da Head, mantendo o DINOv2 intocado.
* **Verificação:** Validaremos o alinhamento das escalas entre o Backbone (Patch Size 14) e as Heads de saída (Resolução 1024x1024).
