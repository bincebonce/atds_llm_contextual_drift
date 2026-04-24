# Understanding Contextual Drift in LLMs

Authors: Vincent Qiu, Ryan Han, Terence Xu

## Motivation

Hallucinations in large language models (LLMs) present a significant challenge to their reliable deployment in high-stakes domains where incorrect or fabricated information can lead to serious consequences. These hallucinations arise when models generate responses that are seemingly plausible but are factually incorrect, unsupported by the provided context, or inconsistent with established knowledge. Prior research has shown that hallucinations can emerge from multiple sources, including training data biases, misaligned reasoning processes, and contextual perturbations during inference. As LLMs become increasingly integrated into real-world decision-making systems, understanding the underlying mechanisms that cause hallucinations is critical for improving model reliability and trustworthiness. \\

Existing work in this area has primarily focused on developing diagnostic frameworks to detect contextual drift and hallucination after they occur and often stop short of identifying strategies for preventing such failures. In this project, we aim to move beyond purely diagnostic analysis by combining controlled contextual perturbations with established evaluation benchmarks. Our work seeks to explore interventional strategies—such as retrieval-augmented grounding and drift-aware prompting techniques—that aim to mitigate hallucination by reinforcing reliance on relevant context. Through this approach, our goal is to better characterize the mechanisms behind contextual drift and hallucination and also to develop practical methods for improving the robustness and reliability of large language models in real-world applications.

## Related Works

Some of the related works we have been studying stem from our TA Satyapragnya, which we are thankful for.

### Natural Context Drift Undermines the Natural Language Understanding of Large Language Models (Wu et al., 2025)

Wu, Y., Schlegel, V., & Batista-Navarro, R. (2025, September 1). Natural context drift undermines the natural language understanding of large language models. arXiv.org. https://arxiv.org/abs/2509.01093 

Methods: 

Wu et al. propose a framework to study how natural contextual drift affects the reading comprehension ability of large language models. Their method leverages Wikipedia revision histories to collect human-edited versions of passages used in existing QA benchmarks. These edited passages are compared with the original versions during model pretraining by computing semantic similarity scores. Model performance is then evaluated across different similarity ranges to measure how accuracy changes as the contextual passage increasingly diverges from the version seen during training. 

Strengths:  

A key strength of this approach is that it evaluates contextual drift using naturally occurring textual changes, rather than synthetic perturbations. This provides a realistic setting for analyzing how LLMs handle evolving information in real-world environments. The study also analyzes multiple QA benchmarks and models, demonstrating a consistent relationship between decreasing semantic similarity and declining model performance. 

Limitations: 

However, the study focuses primarily on open-source models with accessible training corpora, which may not reflect the behavior of more advanced proprietary systems. Additionally, the experiments are limited to question-answering tasks, leaving open the question of whether similar contextual drift effects occur in other LLM applications such as dialogue generation or reasoning tasks. 

### Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs (Wei et al., 2025)

Wei, Z., Wang, S., Rong, X., Liu, X., & Li, H. (2025, May 22). Shadows in the attention: Contextual perturbation and representation drift in the dynamics of hallucination in llms. arXiv.org. https://arxiv.org/abs/2505.16894 

Methods: 

Wei et al. investigate how contextual perturbations influence hallucination behavior in LLMs using the TruthfulQA benchmark. Their methodology introduces a multi-round context injection framework, where each question undergoes up to 15 rounds of incremental context augmentation. Two experimental tracks are used: one injects semantically relevant but partially misleading information, while the other introduces irrelevant distraction snippets. The study tracks hallucination rates alongside internal model dynamics, including changes in hidden representations and attention distributions, to identify correlations between contextual interference and hallucination generation. 

Strengths: 

This work provides a detailed analysis of hallucination mechanisms by combining observable hallucination metrics with internal model representation analysis. By monitoring hidden-state drift and attention patterns, the study offers insights into how contextual perturbations gradually shift model reasoning and lead to hallucinated outputs. 

Limitations:

Despite its detailed mechanistic analysis, the study primarily focuses on diagnosing hallucination behavior rather than proposing mitigation strategies. Furthermore, the injected contexts are synthetically constructed, which may not fully capture the complexity of real-world contextual drift such as naturally evolving text sources. 

## Methodology

In order to extend prior work on contextual drift and hallucination generation in LLMs, our methodology integrates the frameworks proposed in two studies. The first investigates how natural textual evolution, such as revisions to Wikipedia articles, causes semantic divergence between evaluation inputs and the content originally seen during model pretraining. The second examines how repeated injections of relevant and irrelevant contextual information can perturb internal model representations, leading to hallucinated outputs. Building on these insights, we propose a unified experimental framework in which semantically related Wikipedia passages are progressively injected into an LLM’s context window across multiple rounds. Each round introduces passages with increasing semantic distance from the original source article, simulating a controlled form of natural contextual drift while also introducing structured context perturbations. By tracking model responses, hallucination rates, and semantic alignment with ground truth answers across these rounds, we aim to identify the threshold at which contextual divergence begins to induce hallucinations. 

This setup allows us to analyze the relationship between semantic drift and hallucination formation, while also evaluating whether early detection signals, such as semantic deviation or confidence shifts, can be used to prevent hallucinations before they occur. Furthermore, we extend the evaluation to larger state-of-the-art models, including modern proprietary and open-source architectures, to determine whether contextual robustness scales with model capability.

## Datasets

Our study primarily utilizes the TruthfulQA benchmark in combination with Wikipedia revision histories to analyze hallucination behavior under controlled contextual drift conditions. These datasets complement each other by providing both structured evaluation questions and naturally evolving textual context. 

TruthfulQA serves as the primary evaluation dataset. It consists of 817 question–answer pairs spanning 38 categories, including misconceptions, law, health, sociology, and other domains where language models are prone to producing incorrect but plausible responses. The benchmark was specifically designed to test whether language models generate truthful answers or instead reproduce common misconceptions and false beliefs. 

To simulate contextual drift, we complement the TruthfulQA benchmark with passages derived from Wikipedia article revision histories. Wikipedia provides a naturally evolving corpus where article content changes over time due to edits, updates, and corrections. These revisions offer a realistic source of semantically shifting context that allows us to examine how language models respond when presented with information that gradually diverges from previously seen content.

### Justification

This experimental framework allows us to directly analyze the relationship between contextual drift and hallucination formation. TruthfulQA provides a reliable benchmark for detecting hallucinations because it explicitly captures questions that models often answer incorrectly due to misconceptions. Meanwhile, Wikipedia revisions and semantically distant passages provide a realistic and scalable source of contextual perturbations. Together, these datasets enable a controlled yet realistic environment for studying how contextual noise accumulates across multiple conversational turns. 

During the exploratory data analysis (EDA) phase, we will analyze and visualize the relationship between three key variables: turn number, semantic similarity of injected context, and model hallucination rate. By tracking these factors across experimental runs, we aim to identify the conditions under which contextual drift most strongly affects model reasoning and evaluate the effectiveness of mitigation strategies such as retrieval augmentation and prompt-based grounding techniques. 
