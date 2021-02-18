import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

sent_no=10

import torch
print(torch.__version__)

#If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
#
# # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import pandas as pd

df = pd.read_csv("./MELD/data/MELD/train_sent_emo.csv",usecols=['Utterance','Emotion','Dialogue_ID','Speaker'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print('feature cols: {:,}\n'.format(df.shape[1]))

# Display 10 random rows from the data.
print(df.sample(10))

# Display 5 neutral sentences
print(df.loc[df.Emotion == 'neutral'].sample(5)[['Utterance']])

# Extract sentences and labels
sentences = df.Utterance.values
labels = df.Emotion.values
dialogueIDs=df.Dialogue_ID.values
speakers=df.Speaker.values

df = pd.read_csv("./MELD/data/MELD/dev_sent_emo.csv",usecols=['Utterance','Emotion','Dialogue_ID','Speaker'])
val_sentences = df.Utterance.values
val_labels = df.Emotion.values
val_dialogueIDs=df.Dialogue_ID.values
val_speakers=df.Speaker.values

df = pd.read_csv("./MELD/data/MELD/test_sent_emo.csv",usecols=['Utterance','Emotion','Dialogue_ID','Speaker'])
test_sentences = df.Utterance.values
test_labels = df.Emotion.values
test_dialogueIDs=df.Dialogue_ID.values
test_speakers=df.Speaker.values

from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model.cuda()
print("load finish")

# We will use 5 special tokens:
# - <bos> to indicate the start of the sequence
# - <eos> to indicate the end of the sequence
# - <speaker1> to indicate the beginning and the tokens of an utterance from the user
# - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
# - <pad> as a padding token to build batches of sequences
SPECIAL_TOKENS = ["<bos>", "<eos>", '<speaker>',"<speaker1>", "<speaker2>",'<speaker3>','<speaker4>','<speaker5>','<speaker6>','<speaker7>', "<pad>"]

# We can add these special tokens to the vocabulary and the embeddings of the model:
tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))

import numpy as np

# Let's define our contexts and special tokens
# history为含reply(label)的list
# 最多有7个speaker

history = []
speakerList=[]
speakerEmotionDict={} # [0]:neutral [1]:joy [2]:sadness [3]:fear [4]:anger [5]:surprise [6]:disgust
for i,s in enumerate(sentences):
    if speakers[i] not in speakerEmotionDict.keys():
        speakerEmotionDict[speakers[i]]=[0,0,0,0,0,0,0]
    if labels[i]=='neutral':
        speakerEmotionDict[speakers[i]][0]+=1
    elif labels[i]=='joy':
        speakerEmotionDict[speakers[i]][1]+=1
    elif labels[i]=='sadness':
        speakerEmotionDict[speakers[i]][2] += 1
    elif labels[i]=='fear':
        speakerEmotionDict[speakers[i]][3] += 1
    elif labels[i]=='anger':
        speakerEmotionDict[speakers[i]][4] += 1
    elif labels[i]=='surprise':
        speakerEmotionDict[speakers[i]][5] += 1
    elif labels[i]=='disgust':
        speakerEmotionDict[speakers[i]][6] += 1
    else:
        speakerEmotionDict[speakers[i]][0] += 1
    j=i-1
    while j>=0 and i-j<sent_no and dialogueIDs[j]==dialogueIDs[i]:
        j=j-1
    j=j+1
    tempHistory=[]
    tempSpeaker=[]
    tempSpeakerList=[]
    #print(j)
    #print(i+1)
    for k in range(j,i+1):
        tempHistory.append(sentences[k])
        if speakers[k] not in tempSpeaker:
            tempSpeaker.append(speakers[k])
        pos=tempSpeaker.index(speakers[k])
        tempSpeakerList.append('<speaker' + str(pos + 1) + '>')
    tempSpeakerList.append(tempSpeakerList[-1])
    tempHistory.append(labels[i])
    speakerList.append(tempSpeakerList)
    history.append(tempHistory)

for key in speakerEmotionDict.keys():
    sum_value=sum(np.array(speakerEmotionDict[key]))
    speakerEmotionDict[key]=np.array(speakerEmotionDict[key])/sum_value

save_obj(speakerEmotionDict,'speakerEmotionDict')

reply = labels

val_history = []
val_speakerList=[]
for i,s in enumerate(val_sentences):
    j=i
    while j>=0 and i-j<sent_no and val_dialogueIDs[j]==val_dialogueIDs[i]:
        j=j-1
    j=j+1
    tempHistory=[]
    tempSpeaker = []
    tempSpeakerList = []
    for k in range(j,i+1):
        tempHistory.append(val_sentences[k])
        if val_speakers[k] not in tempSpeaker:
            tempSpeaker.append(val_speakers[k])
        pos=tempSpeaker.index(val_speakers[k])
        tempSpeakerList.append('<speaker'+str(pos+1)+'>')
    tempSpeakerList.append(tempSpeakerList[-1])
    tempHistory.append(val_labels[i])
    val_speakerList.append(tempSpeakerList)
    val_history.append(tempHistory)

test_history = []
test_speakerList=[]
final_test_speakers=[]
for i,s in enumerate(test_sentences):
    final_test_speakers.append(test_speakers[i])
    j=i
    while j>=0 and i-j<sent_no and test_dialogueIDs[j]==test_dialogueIDs[i]:
        j=j-1
    j=j+1
    tempHistory=[]
    tempSpeaker = []
    tempSpeakerList = []
    for k in range(j,i+1):
        tempHistory.append(test_sentences[k])
        if test_speakers[k] not in tempSpeaker:
            tempSpeaker.append(test_speakers[k])
        pos=tempSpeaker.index(test_speakers[k])
        tempSpeakerList.append('<speaker'+str(pos+1)+'>')
    tempSpeakerList.append(tempSpeakerList[-1])
    tempHistory.append(test_labels[i])
    test_speakerList.append(tempSpeakerList)
    test_history.append(tempHistory)

sequence=history
val_sequence=val_history
test_sequence=test_history

input_ids=[]
lm_targets=[]
last_tokens=[]
segments=[]
for index,s in enumerate(sequence):#对对话遍历
    tmp_t=['<bos>']
    tmp_s=[speakerList[index][0]]
    for i,ss in enumerate(s):#对每组对话里的句子遍历
        if i!=len(s)-1:#最后一句是情感，不打<speaker>
            tmp_t.append('<speaker>')
            tmp_s.append(speakerList[index][i])
        tmp_tt=tokenizer.tokenize(ss)
        tmp_s+=[speakerList[index][i]]*len(tmp_tt)
        tmp_t=tmp_t+tmp_tt
    tmp_s.append(tmp_s[-1])
    tmp_t.append('<eos>')

    input_id=tokenizer.convert_tokens_to_ids(tmp_t)
    segment=tokenizer.convert_tokens_to_ids(tmp_s)
    last_tokens.append(len(input_id)-1)
    input_ids.append(input_id)
    segments.append(segment)
    lm_target=[-1]*(len(input_id)-2)
    lm_target=lm_target+input_id[-2:]
    lm_targets.append(lm_target)

val_input_ids=[]
val_lm_targets=[]
val_last_tokens=[]
val_segments=[]
for index,s in enumerate(val_sequence):
    tmp_t=['<bos>']
    tmp_s = [val_speakerList[index][0]]
    for i,ss in enumerate(s):
        if i!=len(s)-1:
            tmp_t.append('<speaker>')
            tmp_s.append(val_speakerList[index][i])
        tmp_tt=tokenizer.tokenize(ss)
        tmp_s += [val_speakerList[index][i]] * len(tmp_tt)
        tmp_t=tmp_t+tmp_tt
    tmp_t.append('<eos>')
    val_input_id=tokenizer.convert_tokens_to_ids(tmp_t)
    val_segment = tokenizer.convert_tokens_to_ids(tmp_s)
    val_last_tokens.append(len(val_input_id)-1)
    val_input_ids.append(val_input_id)
    val_segments.append(val_segment)
    val_lm_target=[-1]*(len(val_input_id)-2)
    val_lm_target=val_lm_target+val_input_id[-2:]
    val_lm_targets.append(val_lm_target)

test_input_ids=[]
test_lm_targets=[]
test_last_tokens=[]
test_segments=[]
for index,s in enumerate(test_sequence):
    tmp_t=['<bos>']
    tmp_s = [test_speakerList[index][0]]
    for i,ss in enumerate(s):
        if i!=len(s)-1:
            tmp_t.append('<speaker>')
            tmp_s.append(test_speakerList[index][i])
        tmp_tt=tokenizer.tokenize(ss)
        tmp_s += [test_speakerList[index][i]] * len(tmp_tt)
        tmp_t=tmp_t+tmp_tt
    tmp_t.append('<eos>')
    test_input_id=tokenizer.convert_tokens_to_ids(tmp_t)
    test_segment = tokenizer.convert_tokens_to_ids(tmp_s)
    test_last_tokens.append(len(test_input_id)-1)
    test_input_ids.append(test_input_id)
    test_segments.append(test_segment)
    test_lm_target=[-1]*(len(test_input_id)-2)
    test_lm_target=test_lm_target+test_input_id[-2:]
    test_lm_targets.append(test_lm_target)

print("max length of input_ids:")
print(max(len(s) for s in input_ids))

padding_length=260

def pad(x, padding):
    return x + [padding] * (padding_length - len(x))

input_ids=[pad(x, tokenizer.convert_tokens_to_ids('<pad>')) for x in input_ids]
val_input_ids=[pad(x, tokenizer.convert_tokens_to_ids('<pad>')) for x in val_input_ids]
test_input_ids=[pad(x, tokenizer.convert_tokens_to_ids('<pad>')) for x in test_input_ids]

lm_targets=[pad(x,-1) for x in lm_targets]
val_lm_targets=[pad(x,-1) for x in val_lm_targets]
test_lm_targets=[pad(x,-1) for x in test_lm_targets]

segments=[pad(x,0) for x in segments]
val_segments=[pad(x,0) for x in val_segments]
test_segments=[pad(x,0) for x in test_segments]

input_ids = torch.tensor([input_ids], dtype=torch.long)
lm_labels=torch.tensor([lm_targets],dtype=torch.long)
last_tokens=torch.tensor([last_tokens],dtype=torch.long)
segments = torch.tensor([segments], dtype=torch.long)

val_input_ids = torch.tensor([val_input_ids], dtype=torch.long)
val_lm_labels=torch.tensor([val_lm_targets],dtype=torch.long)
val_last_tokens=torch.tensor([val_last_tokens],dtype=torch.long)
val_segments = torch.tensor([val_segments], dtype=torch.long)

test_input_ids = torch.tensor([test_input_ids], dtype=torch.long)
test_lm_labels=torch.tensor([test_lm_targets],dtype=torch.long)
test_last_tokens=torch.tensor([test_last_tokens],dtype=torch.long)
test_segments = torch.tensor([test_segments], dtype=torch.long)

from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
from transformers import get_linear_schedule_with_warmup

# Number of training epochs.
epochs = 1 # try 2

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).

batch_size=4
import math
total_steps = math.ceil(len(input_ids[0])/batch_size) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import time
import datetime


def format_time(elapsed):
    # Takes a time in seconds and returns a string hh:mm:ss

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42
import numpy as np
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

from sklearn.metrics import f1_score

# For each epoch...
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    i=0
    while i<len(input_ids[0]):
        if (i/batch_size) % 40 == 0 and not i/batch_size == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i/batch_size, math.ceil(len(input_ids[0])/batch_size), elapsed))
        if i+batch_size<len(input_ids[0]):
            temp = input_ids[0][i:i + batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_input_ids=temp.to(device)
            temp = segments[0][i:i + batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_segments = temp.to(device)
            temp = lm_labels[0][i:i + batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_lm_labels=temp.to(device)
            temp = last_tokens[0][i:i + batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_last_tokens=temp.to(device)
        else:
            temp = input_ids[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_input_ids = temp.to(device)
            temp = segments[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_segments = temp.to(device)
            temp = lm_labels[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_lm_labels = temp.to(device)
            temp = last_tokens[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_last_tokens = temp.to(device)

        model.zero_grad()
        lm_loss = model(
                    b_input_ids, lm_labels=b_lm_labels,
                    mc_token_ids=b_last_tokens,token_type_ids=b_segments
                )
        lm_loss=lm_loss[0]

        total_train_loss += lm_loss.item()
        lm_loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        i=i+batch_size

    torch.save(model.state_dict(), './model_save/result'+str(epoch_i)+'.pkl')

    # for validation
    # For each batch of training data...
    print('val begin')
    flat_true_labels=[]
    flat_predictions=[]
    t0 = time.time()
    i = 0
    val_batch_size=16
    total_eval_loss=0
    true_c = 0
    false_c = 0
    while i < len(val_input_ids[0]):
        #print(i/val_batch_size)
        if (i / val_batch_size) % 40 == 0 and not i / val_batch_size == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i / val_batch_size,
                                                                        math.ceil(len(val_input_ids[0]) / val_batch_size),
                                                                        elapsed))
        if i + val_batch_size < len(val_input_ids[0]):
            temp = val_input_ids[0][i:i + val_batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_input_ids = temp.to(device)
            temp = val_segments[0][i:i + val_batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_segments = temp.to(device)
            temp = val_lm_labels[0][i:i + val_batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_lm_labels = temp.to(device)
            temp = val_last_tokens[0][i:i + val_batch_size]
            temp = torch.unsqueeze(temp, dim=1)
            b_last_tokens = temp.to(device)
        else:
            temp = val_input_ids[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_input_ids = temp.to(device)
            temp = val_segments[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_segments = temp.to(device)
            temp = val_lm_labels[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_lm_labels = temp.to(device)
            temp = val_last_tokens[0][i:]
            temp = torch.unsqueeze(temp, dim=1)
            b_last_tokens = temp.to(device)
        with torch.no_grad():
            lm_logits,mc_logits = model(
                b_input_ids,
                mc_token_ids=b_last_tokens,token_type_ids=b_segments
            )
        lm_logits=lm_logits.detach().cpu().numpy()
        for j in range(b_last_tokens.shape[0]):
            id=int(b_last_tokens[j][0].detach().cpu())
            predict=np.argmax(lm_logits[j][0],axis=1)[id-2]
            truth=b_input_ids[j][0][id-1].detach().cpu().numpy()
            flat_true_labels.append(truth)
            flat_predictions.append(predict)
            if predict==truth:
                true_c+=1
            else:
                false_c+=1
        i=i+val_batch_size

    f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
    print('val acc')
    print(true_c/(true_c+false_c))
    print('val f1')
    print(f1)

print('test begin')

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

flat_true_labels=[]
flat_predictions=[]
t0 = time.time()
i = 0
test_batch_size=16
total_eval_loss=0
true_c = 0
false_c = 0
while i < len(test_input_ids[0]):
    if (i / test_batch_size) % 40 == 0 and not i / test_batch_size == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)

        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i / test_batch_size,
                                                                    math.ceil(len(test_input_ids[0]) / test_batch_size),
                                                                    elapsed))
    if i + test_batch_size < len(test_input_ids[0]):
        temp = test_input_ids[0][i:i + test_batch_size]
        temp = torch.unsqueeze(temp, dim=1)
        b_input_ids = temp.to(device)
        temp = test_segments[0][i:i + test_batch_size]
        temp = torch.unsqueeze(temp, dim=1)
        b_segments = temp.to(device)
        temp = test_lm_labels[0][i:i + test_batch_size]
        temp = torch.unsqueeze(temp, dim=1)
        b_lm_labels = temp.to(device)
        temp = test_last_tokens[0][i:i + test_batch_size]
        temp = torch.unsqueeze(temp, dim=1)
        b_last_tokens = temp.to(device)
    else:
        temp = test_input_ids[0][i:]
        temp = torch.unsqueeze(temp, dim=1)
        b_input_ids = temp.to(device)
        temp = test_segments[0][i:]
        temp = torch.unsqueeze(temp, dim=1)
        b_segments = temp.to(device)
        temp = test_lm_labels[0][i:]
        temp = torch.unsqueeze(temp, dim=1)
        b_lm_labels = temp.to(device)
        temp = test_last_tokens[0][i:]
        temp = torch.unsqueeze(temp, dim=1)
        b_last_tokens = temp.to(device)
    with torch.no_grad():
        lm_logits,mc_logits = model(
            b_input_ids,
            mc_token_ids=b_last_tokens,token_type_ids=b_segments
        )
    lm_logits=lm_logits.detach().cpu().numpy()
    for j in range(b_last_tokens.shape[0]):
        id=int(b_last_tokens[j][0].detach().cpu())
        tmp = softmax(lm_logits[j][0][id - 2])
        predict=np.argmax(tmp)
        if final_test_speakers[i+j] in speakerEmotionDict.keys():
            predict_prob=max(tmp)
            tmp[predict]=min(tmp)
            predict_2=np.argmax(tmp)
            predict_2_prob=max(tmp)
            predict_emotion=0
            if predict==11349:
                predict_emotion=0
            elif predict==4505:
                predict_emotion=1
            elif predict==6410:
                predict_emotion=2
            elif predict==1950:
                predict_emotion=3
            elif predict==1721:
                predict_emotion=4
            elif predict==2378:
                predict_emotion=5
            elif predict==6518:
                predict_emotion=6
            predict_emotion_2 = 0
            if predict_2 == 11349:
                predict_emotion_2 = 0
            elif predict_2 == 4505:
                predict_emotion_2 = 1
            elif predict_2 == 6410:
                predict_emotion_2 = 2
            elif predict_2 == 1950:
                predict_emotion_2 = 3
            elif predict_2 == 1721:
                predict_emotion_2 = 4
            elif predict_2 == 2378:
                predict_emotion_2 = 5
            elif predict_2 == 6518:
                predict_emotion_2 = 6
            prev_prob=speakerEmotionDict[final_test_speakers[i+j]][predict_emotion]
            prev_2_prob=speakerEmotionDict[final_test_speakers[i+j]][predict_emotion_2]
            if 0.9*predict_2_prob+0.1*prev_2_prob>0.9*predict_prob+0.1*prev_prob:
                predict=predict_2

        truth=b_input_ids[j][0][id-1].detach().cpu().numpy()
        flat_true_labels.append(truth)
        flat_predictions.append(predict)
        if predict==truth:
            true_c+=1
        else:
            false_c+=1
    i=i+test_batch_size

f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
print('test acc')
print(true_c/(true_c+false_c))
print('test f1')
print(f1)
