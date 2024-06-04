## Introduction
This repository contains several cases study regarding Large Language Model (LLM)

## Performance of fine-tuning methods
There are three main approaches to fine-tune a large language model (LLM): a) updating the model weights via traditional gradient descent, b) using a LoRA adapter, and c) replacing decision layers with domain-specific decision layers while freezing the main body of the LLM. This section investigates the performance of each fine-tuning approach.

##### Experiment Setup

We use Meta OPT-350m, a pre-trained LLM. The LLM is fine-tuned on IMDb, a dataset of 50,000 samples for sentiment analysis. The data is split into 25,000 samples for training and 25,000 samples for testing. In this experiment, we replace the model decision layer with a linear layer of 1,000 neurons to output a binary classification: negative or positive sentiment. We train the model using AdamW on the dataset for three epochs with a batch size of 8. The learning rate is set at 2e-5.

We compare three main approaches: a) traditional training (trad), b) fine-tuning the linear decision layer while the base LLM is frozen (linear), and c) using a low-rank adapter (LoRA).


##### Implementation Detail

* `dev-llm-finetune-sentimental-classification.ipynb` fine-tuning a LLM for sentimental classification on idbm dataset

* `dev-llm-finetune-sentimental-classification-LORA.ipynb` fine-tuning a LLM for sentimental classification on idbm dataset, with LORA adapter

* `dev-llm-finetune-sentimental-classification-Linear-basefrozen-promptengineer.ipynb`, fine-tuning a LLM for sentimental classification on idbm dataset where base model is frozen and a prompt is added for few-shot training

* `dev-llm-finetune-sentimental-classification-Linear-basefrozen.ipynb`, fine-tuning a LLM for sentimental classification on idbm dataset where base model is frozen and only a linear decision layer is trained.

##### Result
As expected, traditional fine-tuning results in the highest accuracy at 92%. The accuracy of LoRA is slightly lower at 89%, as LoRA focuses on fine-tuning the attention component of the model instead of adjusting the parameters of the whole network. We observe that the accuracy of the linear approach is significantly lower compared to the RNN baseline. This result suggests that extra caution should be taken when using the linear approach for fine-tuning.
<img src="https://github.com/phananh1010/llm-use-cases/blob/master/fig_barplot_accuracy.jpg" width="300">


We also measure the inference latency of the three approaches. We see a slight decrease in latency with LoRA compared to traditional fine-tuning. The latency of the linear approach is significantly lower.
<img src="https://github.com/phananh1010/llm-use-cases/blob/master/fig_barplot_latency.jpg" width="300">

