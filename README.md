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
    
2. Store the speaker emotional preference for the training set

    The speakers' names are the keys for the dict, and the values are their emotional preference.
    
    Each value is a list with a length of 7 (7 different emotions), the algorithm tranverses all the emotions of this speaker and the numbers of each emotion are stores them in the list.
    
3. Set specidal tokens

    &lt;bos&gt; &lt;eos&gt; &lt;speaker&gt; &lt;speaker1&gt; &lt;speaker2&gt; &lt;speaker3&gt; &lt;speaker4&gt; &lt;speaker5&gt; &lt;speaker6&gt; &lt;speaker7&gt; &lt;pad&gt;
    
4. Reconstruct the input utterances

    sent_no is set to define how many utterances we will preview. In this code, it is set to 10.
    
    Reconstruct the input utterances according to the sent_no and add the special tokens. 
    
    &lt;bos&gt; &lt;speaker&gt; utterance1 &lt;speaker&gt; utterance2 ... &lt;speaker&gt; utterance10 emotion &lt;eos&gt;
    
5. Segment embeddings

    Do segment embeddings according to the speaker's name. In practice, the number of speakers in our input utterances is no more than 7.
    
    Each token in one speaker's utterance is set to the same segment embedding &lt;speaker1/2/3/4/5/6/7&gt;. Different speakers have different segment embeddings.
    
6. Tokenize, embed and lm_target

    Tokenize the input utterances and add the segment embeddings and position embeddings.
    
    The lm_target (the utterance we want to predict) is set to -1 ... -1 tokenize(emotion) tokenize(&lt;eos&gt;).
    
7. Padding

    In order to ensure each input and lm_target has the same length before they are put into the model, padding is needed. tokenize(&lt;pad&gt;) is added.
    
8. Finetune

    Use Adam optimizer, set the learning rate to be 2e-5, and the epsilon to be 1e-8, train for 1 epoch and the batch size is set to 4.
    
    In each epoch, the model is saved to the dir model_save.
    
    In the validation process, the batch size is set to 16, and the F1-score is printed.

9. Test

    In the test part, the batch size is set to 16. After having obtained the model's output, the two most possible emotions and their possibilities are stored. Then, if the speaker appeared in the training set, his/her emotional preference will be called. 0.9*output possiblility+0.1*preference possibility=final possibility.
