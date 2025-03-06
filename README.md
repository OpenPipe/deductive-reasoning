# Deductive Reasoning

![image/png](https://cdn-uploads.huggingface.co/production/uploads/674a1d102c0f27a385772cfe/JauBmEQM0FpOdShBMSfst.png)

Train your own State-of-the-Art deductive reasoning model with Reinforcement Learning.

## Overview

This repository contains the training recipe for creating state-of-the-art deductive reasoning models using Reinforcement Learning. Our research demonstrates how smaller, open-weight language models can be trained to perform complex logical deduction tasks at frontier-level performance, matching or exceeding proprietary models at a fraction of the cost.

We used the Temporal Clue puzzle dataset to train Qwen 14B and 32B models, improving their deductive reasoning capabilities significantly through Group Relative Policy Optimization (GRPO). Our trained models approach the performance of leading proprietary models like Claude 3.7 Sonnet while maintaining cost efficiency.

## Resources

- **Training Recipe**: This repository (recipe for RL training)
- **Training Dataset**: [Temporal Clue Puzzles](https://github.com/bradhilton/temporal-clue)
- **RL Experiments**: [OpenPipe RL Experiments](https://github.com/openpipe/rl-experiments)
- **Model Weights**:
  - [Deductive Reasoning Qwen 14B](https://huggingface.co/OpenPipe/Deductive-Reasoning-Qwen-14B)
  - [Deductive Reasoning Qwen 32B](https://huggingface.co/OpenPipe/Deductive-Reasoning-Qwen-32B)
- **Blog Post**: [Link to blog post will be added here]

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
