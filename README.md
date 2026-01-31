# Integrating LLM into CI/CD pipeline

## Overview
We have used T5 base encoder-decoder based text summarization model. We are trying to do reinforcement learning by checking on the rouge score at every iteration. If the score is avove a threshold, we choose that summary and display. We are also using Evidently AI monitoring external tool to monitor the LLM app once in production. 
We are pushing all this lot into git repo, running tests on it and if passed push to main branch. This is then deployed in docker container present in docker hub. THat image we can deploy either in standalone serer or in cloud(EC2, etc).

## Tech Stack
- Python on Visual Studio Code
- PyTorch / Transformers
- T5 / LoRA
- Evidently Ai monitoring tool
- Docker Container


## Problem Statement
We are trying to automate the retraining process involved in GenAI models. So everytime the dataset changes at the source, it is well reflected in the production. 

## Approach
- Data preprocessing
- Model / algorithm used is T5 base
- Training strategy ( rouge score, reinforcement learning)
- Evaluation( rouge score, reinforcement learning)

## Results
- ROUGE score achieved was 10


## How to Run
```bash
pip install -r requirements.txt
python llm1.py
python prompty.py
