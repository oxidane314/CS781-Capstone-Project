# In-House Legal Intelligence System
### CS781 Capstone Project — IIT Kanpur

**GitHub Repository:** [CS781 Capstone Project](https://github.com/oxidane314/CS781-Capstone-Project)

## 1. Project Overview

The **In-House Legal Intelligence System** is a multi-task legal NLP system developed as part of the CS781 Capstone Project. It explores the application of language models and parameter-efficient adaptation techniques to multiple legal-domain natural language processing tasks.

The project is organised into seven phases. It begins with training legal-domain language models from scratch and subsequently uses these models as backbones for downstream legal NLP tasks, including named entity recognition, rhetorical role prediction, judgment prediction, statute identification, summarisation and prior-case retrieval.

The central objective is to build a reusable modelling pipeline in which a common set of legal language model backbones can support different legal intelligence tasks through fine-tuning, lightweight adaptation, hierarchical modelling and knowledge distillation.

## 2. Project Structure

The implementation is organised into seven phases:

| Phase | Task | Objective |
|---|---|---|
| 1 | Legal Language Modelling | Train language models from scratch on legal text. |
| 2 | Legal Named Entity Recognition | Identify and classify legal entities and spans. |
| 3 | Rhetorical Role Prediction | Classify sentences according to their rhetorical roles in legal documents. |
| 4 | Court Judgment Prediction | Predict the binary outcome of a court judgment. |
| 5 | Legal Statute Identification | Identify relevant statutes in legal cases using multi-label classification. |
| 6 | Abstractive Legal Summarisation | Generate concise summaries of legal judgments. |
| 7 | Prior Case Retrieval | Retrieve relevant prior cases for a given legal query. |

## 3. Dataset

The project uses the datasets provided for the CS781 Capstone tasks.

**Dataset repository:** [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)

The notebook retrieves the required phase-specific files from Hugging Face using `huggingface_hub` and the `datasets` library.

The data is organised by task and split into training, development/validation and test sets, wherever provided.

The implementation uses the provided splits for model training, validation, experiment comparison and final prediction generation.

## 4. Technical Architecture and Methodology

### Phase 1 — Legal Language Modelling

**Objective:** Build legal-domain language models from scratch that can serve as reusable backbones for subsequent tasks.

**Architectures explored:**
- GPT-2
- LLaMA-style decoder architecture
- RetNet
- Mixture-of-Experts (MoE)

**Methodology:**
- Prepare and tokenize the legal text corpus.
- Train models using next-token prediction.
- Use cached tokenized data and resumable checkpoints.
- Compare architectures using validation loss and per-word perplexity.
- Perform decoding configuration experiments on held-out prompts.
- Generate 128-word continuations for the final test prompts.

**Implementation:** `cs781_phase1.py`

### Phase 2 — Legal Named Entity Recognition

**Objective:** Detect legal entities and identify their corresponding labels in legal documents.

**Methodology:**
- Use the Phase 1 language models as pretrained backbones.
- Process legal text while preserving token-to-character offset mappings.
- Incorporate document-position and line-break features.
- Compare full fine-tuning and parameter-efficient adaptation approaches.
- Use a token-level NER prediction head.
- Explore knowledge distillation using a LLaMA teacher and a pruned RetNet student.
- Evaluate predictions using strict span-level macro-F1.

**Adaptation approaches explored:**
- Full fine-tuning
- LoRMA+
- LoRA
- Knowledge distillation

**Implementation:** `cs781_phase2_ner.py`

### Phase 3 — Rhetorical Role Prediction

**Objective:** Classify sentences in legal documents according to their rhetorical roles across the CL and IT domains.

**Methodology:**
- Segment documents into sentences.
- Encode sentences using Phase 1 language model backbones.
- Incorporate document-level context through a BiGRU over sentence representations.
- Predict rhetorical roles using a classification head.
- Compare contextual document-level prediction against sentence-only classification.
- Explore parameter-efficient adaptation and knowledge distillation.

**Techniques explored:**
- LoRMA+
- Rank-Stabilised LoRA (rsLoRA)
- IA3
- Full fine-tuning
- Knowledge distillation

**Implementation:** `cs781_phase3_rr.py`

### Phase 4 — Court Judgment Prediction

**Objective:** Predict the binary outcome of a court judgment from its textual content.

**Methodology:**
- Split long judgments into chunks to handle documents exceeding the model context window.
- Experiment with different chunk-selection strategies:
  - Tail
  - Head
  - Head-tail
  - Uniform
- Encode selected chunks using Phase 1 backbones.
- Aggregate chunk representations using hierarchical pooling mechanisms.
- Combine pooled representations with document-position information.
- Use an MLP classification head for outcome prediction.
- Tune the classification threshold using development data.
- Compare hierarchical aggregation strategies and adaptation methods.

**Techniques explored:**
- Full fine-tuning
- LoRMA+
- Houlsby adapters
- Knowledge distillation

**Implementation:** `cs781_phase4_cjpe.py`

### Phase 5 — Legal Statute Identification

**Objective:** Identify the statutes relevant to a given legal case as a multi-label classification problem.

**Methodology:**
- Divide long legal judgments into manageable token chunks.
- Sample chunks during training and distribute chunk selection across the document during inference.
- Use label-wise attention, where each statute label has its own attention query over the document chunks.
- Apply weighted Binary Cross-Entropy to address class imbalance.
- Explore global, per-label and cross-fitted decision-threshold strategies.
- Generate unique statute IDs for each test case.

**Techniques explored:**
- Full fine-tuning
- LoRMA+
- Houlsby adapters
- Knowledge distillation

**Implementation:** `cs781_phase5_lsi.py`

### Phase 6 — Abstractive Legal Summarisation

**Objective:** Generate abstractive summaries of legal judgments while retaining their important information.

**Methodology:**
- Use the Phase 1 decoder-only language models as the summarisation backbone.
- Select important source sentences using an importance-based selector.
- Preserve the original order of selected sentences.
- Fine-tune the language model using source-summary training pairs.
- Apply loss masking so that the summarisation objective is applied to summary tokens.
- Explore different adaptation approaches and decoding configurations.
- Use the full fine-tuned LLaMA model as the teacher for the RetNet student.

**Systems explored:**
- Full fine-tuning with LLaMA
- LoRMA+ with GPT-2
- Houlsby adapters with MoE
- Knowledge distillation with RetNet

**Implementation:** `cs781_phase6_summ.py`

### Phase 7 — Prior Case Retrieval

**Objective:** Retrieve relevant prior cases for a given legal query.

**Methodology:**
- Implement a dual-encoder retrieval architecture using shared Phase 1 language model backbones.
- Split long cases into token chunks and generate chunk-level representations.
- Aggregate chunk representations using different pooling strategies.
- Train using contrastive InfoNCE loss with in-batch negatives.
- Mine hard negatives using BM25.
- Explore neural similarity and lexical retrieval scores.
- Fuse neural and BM25 scores using a development-tuned weighting strategy.
- Generate seven unique candidate case IDs for each query.

**Techniques explored:**
- Full fine-tuning
- LoRMA+
- IA3
- Knowledge distillation

**Implementation:** `cs781_phase7_pcr.py`

## 5. Common Modelling Components

The project reuses the Phase 1 language model backbones across downstream tasks.

The following techniques are explored across different phases:

| Technique | Purpose |
|---|---|
| Full fine-tuning | Update the complete model for a downstream task. |
| LoRA | Introduce trainable low-rank updates while keeping the backbone frozen. |
| LoRMA+ | Explore multiplicative low-rank adaptation. |
| rsLoRA | Use rank-stabilised scaling for low-rank adaptation. |
| IA3 | Adapt models using learned scaling vectors. |
| Houlsby adapters | Insert lightweight bottleneck modules into model layers. |
| Knowledge distillation | Transfer knowledge from a larger teacher model to a smaller student model. |
| BM25 | Provide a lexical retrieval baseline and hard-negative mining. |

The specific techniques and task-specific prediction heads vary by phase.

## 6. Implementation and Execution

The main implementation is provided in the Jupyter notebook under `notebooks/`.

The notebook contains:
- Environment and dependency setup.
- Python implementation scripts for individual phases.
- Self-tests and implementation checks.
- Dataset preparation and validation.
- Model training and checkpoint handling.
- Experiment configurations and comparisons.
- Prediction generation.
- Submission ZIP creation.

### Execution Environment

The experiments are designed primarily for a GPU-enabled Kaggle environment.

The notebook uses cached intermediate data and resumable checkpoints to avoid repeating completed stages unnecessarily.

### General Execution Workflow

1. Open the notebook in a GPU-enabled environment.
2. Install the required dependencies using the setup cells.
3. Authenticate with Hugging Face when downloading the dataset requires authentication.
4. Execute the Phase 1 preparation, self-test and training stages.
5. Run the downstream phases using the Phase 1 checkpoints.
6. Execute the task-specific self-tests and data checks before training.
7. Compare experiment results using the development metrics.
8. Generate the final predictions and submission ZIP files.

**Important:** Downstream phases depend on the Phase 1 model checkpoints and shared implementation files. Follow the execution order and dependencies documented in the notebook.

## 7. Evaluation

Each phase uses task-appropriate evaluation metrics.

| Phase | Evaluation |
|---|---|
| Phase 1 | Validation loss, per-word perplexity and text-generation metrics. |
| Phase 2 | Strict span-level macro-F1. |
| Phase 3 | Rhetorical role macro-F1. |
| Phase 4 | Binary classification macro-F1. |
| Phase 5 | Multi-label classification metrics, including macro-F1 and micro-F1. |
| Phase 6 | BERTScore and ROUGE metrics. |
| Phase 7 | Micro-F1@7, along with retrieval precision and recall. |

Development-set metrics are used for local model comparison and experiment analysis. Official evaluation scores are determined separately by the course evaluation platform.

## 8. Outputs and Submission Files

Each phase generates predictions in the format required by its corresponding task.

The notebook includes functionality to:
- Save model checkpoints.
- Record experiment configurations and metrics.
- Generate prediction files.
- Validate output formats.
- Package predictions into submission ZIP files.
- Collect required submission files into phase-specific folders.

The generated submission files are kept separate from the source implementation.

## 9. Repository Contents

```text
CS781-Capstone-Project/
│
├── README.md
│
├── notebooks/
│   └── CS781_Capstone.ipynb
│
├── src/
│   ├── cs781_phase1.py
│   ├── cs781_phase2_ner.py
│   ├── cs781_phase3_rr.py
│   ├── cs781_phase4_cjpe.py
│   ├── cs781_phase5_lsi.py
│   ├── cs781_phase6_summ.py
│   └── cs781_phase7_pcr.py
│
└── .gitignore
```

The notebook contains the complete phase-wise workflow. The `src/` directory provides standalone copies of the implementation scripts for easier inspection and reuse.

## 10. Acknowledgements

This project was developed as part of the CS781 Capstone Project at the Indian Institute of Technology Kanpur.

The work uses the datasets and task definitions provided for the course, along with open-source machine learning frameworks and libraries.

---

**Project:** In-House Legal Intelligence System  
**Course:** CS781 Capstone Project  
**Institution:** Indian Institute of Technology Kanpur
