# Reddit_Tag_and_Title_Prediction

### Overview

This project focuses on leveraging transformer models for the task of predicting tags and titles. The implementation involves two main components: **Data Preparation** and **Tag and Title Prediction**. The former covers all the necessary preprocessing tasks required for model training and evaluation, while the latter involves training a transformer model using the `simpletransformers` library and the T5Model class, fine-tuned with a custom dataset.

### Table of Contents

1. [Data Preparation](#data-preparation)
   - [Overview](#overview)
   - [Usage](#usage)
   - [Dependencies](#dependencies)

2. [Tag and Title Prediction](#tag-and-title-prediction)
   - [Overview](#overview)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
   - [Dependencies](#dependencies)

### Data Preparation

#### Overview

The **data preparation** component is crucial for creating a reliable dataset that can be used for training and evaluating the transformer model. It encompasses tasks such as data cleaning, tokenization, and any other preprocessing steps required to format the data appropriately.
Notebook perform all the necessary step for model training and convert it into required format according to the model.

### Dependencies
- pandas
- numpy
- seaborn
- ast
- sklearn
- simpletransformers
- wandb

### Tag and Title Prediction

#### Overview
The **tag and title prediction** component involves training a transformer model using the simpletransformers library, specifically utilizing the T5Model class. The model is fine-tuned with a custom dataset created during the data preparation phase.
You can see the [T5Model docs](https://simpletransformers.ai/docs/t5-model/) for more details about the model.you can see the [model args](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model) if you want to explore about them and understand what each of them is doing.

#### What is Transformers ?
A Transformer is a type of deep learning architecture that uses an attention mechanism to process text sequences. Unlike traditional models based on recurrent neural networks, Transformers do not rely on sequential connections and are able to capture long-term relationships in a text.
Transformer models are a type of deep learning model that is used for natural language processing (NLP) tasks. They are able to learn long-range dependencies between words in a sentence, which makes them very powerful for tasks such as machine translation, text summarization, and question answering.

#### What is fine-tuning in Transformers?
Fine-tuning is a process in which a pre-trained model is further trained on a new task using task-specific data. In the context of Transformer models, fine-tuning refers to the process of using a pre-trained Transformer model as the starting point for training on a new task.
The idea behind fine-tuning Transformer models is that they have already been trained on a large corpus of text data, and therefore have already learned many useful representations of language. By fine-tuning the model on a new task, the model can use these pre-learned representations as a good starting point, and learn task-specific information from the new task data.
The process of fine-tuning a Transformer model involves unfreezing some or all of the layers of the pre-trained model and training them on the new task data using a task-specific loss function. The remaining layers can be kept frozen, preserving the pre-learned representations and preventing overfitting on the small task-specific data.

  
