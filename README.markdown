# Fine-Tuning QWen 2.5 1.5B-Instruct with SFT and DPO

## Introduction

This project demonstrates the fine-tuning of the QWen 2.5 1.5B-Instruct model using two advanced techniques: Supervised Fine-Tuning (SFT) with QLoRA and Direct Preference Optimization (DPO) with LoRA. The objective is to train the model to accurately classify whether given labels are sensitive in various languages, using a dataset from Hugging Face's ai4privacy.

## Dataset

The dataset used in this project is sourced from Hugging Face's ai4privacy datasets, which focus on personally identifiable information (PII). Specifically, we used a custom dataset extracted from these resources, named `PII_dataset_2.xlsx`. This dataset contains labels in different languages along with their sensitivity status, which is used to train and evaluate the model.

## Methodology

### SFT with QLoRA

- **Model**: [QWen 2.5 1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Quantization**: 4-bit quantization using `BitsAndBytesConfig`
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 16
  -26  - Target modules: `q_proj`, `v_proj`
  - Dropout: 0.1
- **Training**:
  - Batch size: 2 (train), 4 (eval)
  - Epochs: 3
  - Learning rate: 2e-4
  - Optimizer: `paged_adamw_8bit`
  - Evaluation every 50 steps
  - Save at the end of each epoch

### DPO with LoRA

- **Starting Model**: SFT checkpoint
- **Preference Pairs**:
  - Chosen: "yes" if label is sensitive, "no" otherwise
  - Rejected: opposite of chosen
- **Training**:
  - Batch size: 2 (train), 4 (eval)
  - Epochs: 3
  - Learning rate: 2e-4
  - Other parameters similar to SFT

## Results

The performance of the Original, SFT, and DPO models was evaluated on the test set. The metrics include accuracy, precision, recall, and F1 score. For detailed results, please refer to the bar plot below.

![Results](image.png)

## How to Run

To run this project, you need to have the following dependencies installed:

- Python 3.8+
- Transformers
- PEFT
- Datasets
- TRL
- Scikit-learn
- Tqdm
- Torch (with CUDA support)

You can run the Jupyter notebook on Google Colab or locally if you have the necessary setup.

1. Upload the dataset file `PII_dataset_2.xlsx` to your environment.
2. Run the notebook cells in order.

## References

- **QWen 2.5 Model**: [QWen 2.5 1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **ai4privacy Dataset**: [ai4privacy on Hugging Face](https://huggingface.co/datasets/ai4privacy)
- **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **DPO Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
