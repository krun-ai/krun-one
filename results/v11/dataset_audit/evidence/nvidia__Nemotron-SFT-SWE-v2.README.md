---
language:
- en
license:
- cc-by-4.0
- apache-2.0
- mit
- bsd-3-clause
- bsd-2-clause
task_categories:
- text-generation
tags:
- tool-use
- SWE
configs:
- config_name: default
  data_files:
  - split: agentless
    path: data/agentless.jsonl
  - split: openhands_swe
    path: data/swe.jsonl
---

## Dataset Description:
Nemotron-SWE-v2 is a software engineering instruction tuning dataset designed to advance the capabilities of LLMs on SWE-Bench style tasks. It includes both agentic trajectories collected using the OpenHands framework and an agentless SWE subset for targeted sub-tasks such as code localization, code repair, and test generation.

This dataset is ready for commercial use.

## Dataset Subsets:
Nemotron-SWE-v2 contains the following subsets:

### Agentic SWE trajectories
This subset comprises ~46k agent trajectories collected using the OpenHands framework. The trajectories were synthesized using state-of-the-art Qwen3-Coder-480B-A35B-Instruct and specifically curated for supervised fine-tuning (SFT), aiming to improve model performance on SWE-Bench style tasks. The issue statements are sourced from SWE-Gym and R2E-Gym-Subset (prompts are used to generate problem statements using Qwen3-Coder-480B-A35B-Instruct).

### Agentless SWE
Agentless SWE is supervised fine-tuning data for agentless software engineering sub-tasks in a SWE-Bench style setting: code localization, code repair, and test generation. Each example is formatted with task-specific outputs such as a ranked file list for localization, patch edits for repair, and reproduction plus unit tests for test generation.

SFT dataset generation: DeepSeek-R1-0528 produces multiple candidate outputs per prompt (8 responses for SWE-Bench-Train / SWE-reBench / SWE-Smith and 4 responses for SWE-Fixer-Train). Prompts include the task specification and issue or problem statement. For repair prompts also include the relevant code files.



## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 12/01/2025  
Last Modified on: 12/01/2025


## License/Terms of Use: 
This dataset is licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).  
Additional Information: Apache 2.0 License; MIT License; BSD-3 License; BSD-2 License.

## Intended Usage:

This dataset is intended for LLM engineers and research teams building autonomous software engineering agents and code-focused assistants. It is suitable for supervised fine-tuning and distillation of models that must interpret real-world issue statements, plan multi-step tool use, navigate codebases, and implement fixes in a SWE-Bench–style setting. The trajectories can also be used to benchmark and debug agent policies, improve repository-aware reasoning, and study robust, regression-free code editing behaviors in both academic and production environments.

## Dataset Characterization
**Data Collection Method**  
Hybrid: Automated, Synthetic  

**Labeling Method**
Hybrid: Automated, Synthetic  

## Dataset Format  
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification

| Subset | Samples |
|--------|---------|
| agentless_swe | 209,976 |
| openhands_swe       | 46,278 |
| Total            | 256,254 |

Total Data Storage: ~17GB


## Reference(s):

* [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)
* [R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents](https://arxiv.org/abs/2504.07164)
* [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](https://arxiv.org/abs/2505.20411)
* [The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](https://arxiv.org/abs/2511.03690)
* [Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models](https://arxiv.org/pdf/2512.13607)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)

