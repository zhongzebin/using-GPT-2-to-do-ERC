# using-GPT-2-to-do-ERC
Abstract: using GPT-2 with segment embeddings and speaker related infomation to do emotion recognition in conversations on MELD

Environment: Windows 10, Python 3.7, cuda 11, pytorch 1.7

Pipeline: OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

Reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

Performance: our f1 result is better than DialogueGCN but worse than COSMIC (state-of-the-art)

How to run this project?

1. configure the environment
2. make the dir named model_save (where the fine-tuned model will be saved) in the root dir of your project
3. run finetuneGPT.py

