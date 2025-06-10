# ftw2v_fr_healthcare_v1

**Official repository for the paper**:  
**["A Study on the Relevance of Generic Word Embeddings for Sentence Classification in Hepatic Surgery"](https://ieeexplore.ieee.org/abstract/document/10479342)**

## 🧾 Overview

This repository provides code and pretrained word embedding models designed for French biomedical text processing, introduced in our study. It supports:

- A custom **Word2Vec** model: `fr_w2v.pkl`
- A **FastText-enhanced Word2Vec** model: `fr_w2v_fasttext.pkl`

These models are intended for tasks such as sentence embedding, classification, and similarity measurement in clinical contexts.

---

## 🧠 Main Finding and Abstract

While the fine-tuning process of extensive contextual language models often demands substantial computational capacity, utilizing generic pre-trained models in highly specialized domains can yield suboptimal results. This paper aims to explore an innovative approach to derive pertinent word embeddings tailored to a specific domain with limited computational resources (The introduced methodologies are tested within the domain of hepatic surgery, utilizing the French language.). This exploration takes place within a context where computational limitations prohibit the fine-tuning of large language models.

A new embedding (referred to as FTW2V) that combines Word2Vec and FastText is introduced. This approach addresses the challenge of incorporating terms absent from Word2Vec’s vocabulary. Furthermore, a novel method is used to evaluate the significance of word embeddings within a specialized corpus. This evaluation involves comparing classification scores distributions of classifiers (Gradient Boosting) trained on word embeddings derived from benchmarked Natural Language Processing (NLP) models.

As per this assessment technique, the FTW2V model, trained from scratch with limited computational resources, outperforms generic contextual models in terms of word embeddings quality. Additionally, a computationally efficient contextual model rooted in FTW2V is introduced. This modified model substitutes Gradient Boosting with a transformer and integrates Part Of Speech labels.

---

## 📦 Setup (via Conda)

Clone the repository and create the environment using the provided `environment.yml`:

```bash
git clone https://huggingface.co/AchOk78/ftw2v_fr_healthcare_v1
cd ftw2v_fr_healthcare_v1
conda env create -f environment.yml
conda activate ftw2v_env