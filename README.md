## Introduction
This repository contains several cases study regarding Large Language Model (LLM)

## Performance of fine-tuning methods
There are three main approaches to fine-tune an LLM: a) updating the model weights via traditional gradient descent, b) using a LoRA adapter, and c) replacing decision layers with domain-specific decision layers while freezing the main body of the LLM. This section investigate the performance of each fine-tuning approach. 

##### Experiment Setup
We use Meta OPT-350m, a pre-trained LLM. The LLM is trained on idbm, a dataset of 50,000 samples for sentimental analysis. The data is split, 25,000 sampels for training and 25,000 samples for testing. In this experiment, we replace the model decision layer with a linear layer of 1000 neurons to output a binary classification: negative of positive sentiment. We train the model using AdamW on the dataset for three epochs with batch size of 8. The learning rate is set at 2e-5.

We compare three main approaches, a) trad: traditional training, b) linear:fine-tuning linear decision layer while the base LLM is frozen, and c) LORA: use lowrank adapter.


##### Implementation Detail

* `dev-llm-finetune-sentimental-classification.ipynb` fine-tuning a LLM for sentimental classification on idbm dataset

* `dev-llm-finetune-sentimental-classification-LORA.ipynb` fine-tuning a LLM for sentimental classification on idbm dataset, with LORA adapter

* `dev-llm-finetune-sentimental-classification-Linear-basefrozen-promptengineer.ipynb`, fine-tuning a LLM for sentimental classification on idbm dataset where base model is frozen and a prompt is added for few-shot training

* `dev-llm-finetune-sentimental-classification-Linear-basefrozen.ipynb`, fine-tuning a LLM for sentimental classification on idbm dataset where base model is frozen and only a linear decision layer is trained.

##### Result
As expected, traditional finetuning result in the highest accuracy, at 92%. The accuracy of LORA is a little bit lower, at 89%, as LORA focusing on finetuning the attention component of the model instead of adjusting the parameters of the whole network. We observe the accuracy of linear approach is significantly lower when compared to RNN baseline. This result suggest that extra caution should be paid when using linear approach for fine tuning.

![Bar plot accuracy](https://github.com/phananh1010/llm-use-cases/blob/master/fig_barplot_accuracy.pdf)

We also measure the inference latency of the there approaches. We see a slight decrease in latency in LORA compared to traditional fine-tuning. Latency of Linear is significantly lower.  

![Bar plot accuracy](https://github.com/phananh1010/llm-use-cases/blob/master/fig_barplot_latency.pdf)
