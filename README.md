# using-GPT-2-to-do-ERC
Abstract: using GPT-2 with segment embeddings and speaker related infomation to do emotion recognition in conversations on MELD

Environment: Windows 10, Python 3.7, cuda 11, pytorch 1.7

Pipeline: OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

Reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

Dataset: https://github.com/declare-lab/MELD

Performance: our f1 result is better than DialogueGCN but worse than COSMIC (state-of-the-art)

How to run this project?

1. configure the environment
2. make the dir named model_save (where the fine-tuned model will be saved) in the root dir of your project
3. run finetuneGPT.py

Steps:

1. Load the dataset and pipeline model

    We need to load the cols named Utterance, Emotion, Dialogue_ID, and Speaker in the csv file.
    
    When loading pipeline models, it may take a long time at the first time.
    
2. Set specidal tokens

    "<bos> <eos> <speaker> <speaker1> <speaker2> <speaker3> <speaker4> <speaker5> <speaker6> <speaker7> <pad>"
    
3. Reconstruct the input utterances

    sent_no is set to define how many utterances we will preview. In this code, it is set to 10.
    
    Reconstruct the input utterances according to the sent_no and add the special tokens. 
    
    <bos> <speaker> utterance1 <speaker> utterance2 ... <speaker> utterance10 <eos>
    
4. Segment embeddings

    Do segment embeddings according to the speaker's name. In practice, the number of speakers in our input utterances is no more than 7.
    
    Each token in one speaker's utterance is set to the same segment embedding <speaker1/2/3/4/5/6/7>. Different speakers have different segment embeddings.
