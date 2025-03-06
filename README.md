# Deductive Reasoning

![image/png](https://cdn-uploads.huggingface.co/production/uploads/674a1d102c0f27a385772cfe/JauBmEQM0FpOdShBMSfst.png)

Train your own frontier-level deductive reasoning model with reinforcement learning.

## Overview

This repository contains the training recipe for creating frontier-level deductive reasoning models using reinforcement learning. Our research demonstrates how smaller, open-weight language models can be trained to perform complex logical deduction tasks at frontier-level performance, matching or exceeding proprietary models at a fraction of the cost.

We used the Temporal Clue puzzle dataset to train Qwen 14B and 32B models, improving their deductive reasoning capabilities significantly through Group Relative Policy Optimization (GRPO). Our trained models approach the performance of leading proprietary models like Claude 3.7 Sonnet while maintaining cost efficiency.

## Resources

- **Training Recipe**: This repository (recipe for RL training)
- **Training Dataset**: [Temporal Clue Puzzles](https://github.com/bradhilton/temporal-clue)
- **RL Experiments**: [OpenPipe RL Experiments](https://github.com/openpipe/rl-experiments)
- **Model Weights**:
  - [Deductive Reasoning Qwen 14B](https://huggingface.co/OpenPipe/Deductive-Reasoning-Qwen-14B)
  - [Deductive Reasoning Qwen 32B](https://huggingface.co/OpenPipe/Deductive-Reasoning-Qwen-32B)
- **Blog Post**: [Link to blog post will be added here]

## Getting Started

Follow these steps to run the training recipe:

### Prerequisites

- NVIDIA GPUs with sufficient VRAM for your chosen model:
  - Qwen 14B requires at least 2 GPUs
  - Qwen 32B requires at least 4 GPUs
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bradhilton/deductive-reasoning.git
   cd deductive-reasoning
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. (Optional) Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to add your Weights & Biases API key and project name:

   ```
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_PROJECT=your_project_name
   ```

### Running the Training

1. Open the `train.ipynb` notebook or `train.py` script and configure the training parameters:

   - Set a unique `run_name` for your experiment
   - Choose the model (e.g., `models.qwen_14b()` or `models.qwen_32b()`)
   - Adjust other parameters as needed (learning rate, number of iterations, etc.)

2. Run the training:

   - If using the notebook: Execute all cells in `train.ipynb`
   - If using the script: Run `python train.py`

3. Monitor training progress in Weights & Biases.

The training process will save the latest and/or best checkpoints in your output directory, allowing you to resume training if interrupted.

## Methodology

Our training approach used reinforcement learning to incrementally improve models' deductive reasoning capabilities:

1. **Environment**: Temporal Clue puzzles (inspired by the board game Clue/Cluedo) with verifiable solutions
2. **Algorithm**: Group Relative Policy Optimization (GRPO) without KL divergence penalty
3. **Training Loop**:
   - Generate model responses to puzzle tasks
   - Grade responses and estimate advantages for each group of completions
   - Fine-tune the model using clipped policy gradients
   - Repeat with new puzzles until peak performance

We used the torchtune library for efficient training and vLLM for inference, with the following key parameters:

- Models: Qwen 2.5 Instruct 14B & 32B
- Tasks per Iteration: 32
- Samples per Task per Iteration: 50
- Learning Rate: 6e-6

## Results

Our training produced impressive performance gains, demonstrating that open-weight models can achieve frontier-level reasoning capabilities.

![image](https://github.com/user-attachments/assets/c405846e-3f19-4b0e-a4ac-02f16c015c54)

We dramatically improved the cost-accuracy tradeoff compared to proprietary models:

![image](https://github.com/user-attachments/assets/5889e53e-7d11-4742-900d-5386aadc1983)

Notably, we discovered that meaningful performance improvements (10-15%) can be achieved with as few as 16 training examples, making this approach accessible even with limited data.

## License

This training recipe is freely available under the MIT license.
