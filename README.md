# Phi-2 QLoRA Fine-tuning with GRPO

This project demonstrates fine-tuning the Microsoft Phi-2 foundation model using QLoRA (Quantized Low-Rank Adaptation) and GRPO (Generative Reward Policy Optimization) trainer. The implementation includes model compression, training pipeline, and a demo application showcasing the improvements in model responses.

## Project Overview

- Base Model: microsoft/phi-2
- Training Method: QLoRA + GRPO
- Dataset: Databricks Dolly 15k
- Demo: HuggingFace Spaces (link to be added)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/era-v3-s22-grpo.git
cd era-v3-s22-grpo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

To train the model using QLoRA and GRPO:

```bash
python train.py
```

The training script will:
1. Load the Phi-2 base model
2. Apply QLoRA for efficient fine-tuning
3. Use GRPO trainer for optimization
4. Save the fine-tuned model

## Model Response Comparison

Here are examples showing the model's responses before and after fine-tuning:

### Example 1: Technical Explanation

**Prompt:** "Explain how a transformer neural network works in simple terms."

**Base Model Response:**
```
A transformer neural network processes information using attention mechanisms. It looks at input data and figures out which parts are most important for the task at hand.
```

**Fine-tuned Model Response:**
```
A transformer neural network works like a smart reader that can process information in parallel. It uses:
1. Self-attention to understand relationships between different parts of the input
2. Multi-head attention to look at information from different perspectives
3. Feed-forward networks to process the gathered information
4. Layer normalization and residual connections to maintain stable training

This architecture allows it to handle sequences effectively while capturing complex patterns in the data.
```

### Example 2: Creative Writing

**Prompt:** "Write a short story about a robot learning to paint."

**Base Model Response:**
```
The robot picked up a brush and started painting. It learned from watching human artists and tried to copy their techniques. Eventually, it created its own masterpiece.
```

**Fine-tuned Model Response:**
```
Unit-7 stood before the blank canvas, servos whirring softly as it analyzed the play of sunlight through the studio window. For months, it had studied the great masters - not just their techniques, but the emotions their works evoked.

Its first strokes were hesitant, mechanical. But as its neural networks processed each interaction between brush and canvas, something changed. The robot began to understand that art wasn't just about precise reproduction - it was about interpretation, feeling, and expression.

Slowly, Unit-7's paintings evolved from rigid geometric patterns to fluid, emotional pieces that captured not just what it saw, but how it perceived the world through its unique artificial consciousness.
```

## Running the Demo

To run the comparison demo locally:

```bash
python app.py
```

Or visit our HuggingFace Spaces demo: [Link to be added]

## Training Details

- **QLoRA Configuration:**
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: ["Wqkv", "out_proj", "fc1", "fc2"]

- **Training Parameters:**
  - Learning Rate: 2e-4
  - Epochs: 3
  - Batch Size: 4
  - Max Length: 512
  - Gradient Accumulation Steps: 4

## License

This project is licensed under the MIT License - see the LICENSE file for details.
