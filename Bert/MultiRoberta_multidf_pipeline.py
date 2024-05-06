#!/usr/bin/env python
# coding: utf-8

# ### Data Preprocessing

# In[1]:


import numpy as np
import pandas as pd
import re


# In[2]:


dataset_location = "/Users/Xutao/Documents/CR4CR/dataset/"


# In[3]:


df = pd.read_excel(dataset_location + "Market 00abc_240310.xlsx")


# In[4]:


df['Administration'].unique()


# In[5]:


# Since there are some null responses will a score of 0, we want to replace them with empty strings
df.fillna("", inplace=True)

# select only the three responses columns and the score column
response_columns = ["Market.00a_OE", "Market.00bc_OE", "Market.00bc_OE follow up"]
score_column = ["Score"]
df = df[["Respondent Id", "Administration"] + response_columns + score_column]


# In[6]:


def preprocess_text(text):
    # Lowercase the text
    # text = text.lower()

    text = re.sub(r'\n', ' ', text)
    #text = re.sub(r'^\w+\s*$', '',text)

    text = re.sub(r'[^a-zA-Z0-9\+\-\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text

def preprocess_text1(text):
    # Lowercase the text
    text = text.lower()

    # text = re.sub(r'\n', ' ', text)
    # Remove special characters
    #text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

# Preprocess the text in the response columns
for column in response_columns:
    df[column] = df[column].astype(str).apply(preprocess_text1)


# ### Create a MultiRoberta Model

# In[7]:


import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch import nn

class MultimodalRoberta(torch.nn.Module):
    def __init__(self, num_labels=5):
        super(MultimodalRoberta, self).__init__()
        self.num_labels = num_labels
        self.roberta1 = RobertaModel.from_pretrained('roberta-base')
        self.roberta2 = RobertaModel.from_pretrained('roberta-base')
        self.roberta3 = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Linear(self.roberta1.config.hidden_size + self.roberta2.config.hidden_size + self.roberta3.config.hidden_size, num_labels)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, input_ids_c, attention_mask_c):
        output_a = self.roberta1(input_ids=input_ids_a, attention_mask=attention_mask_a)
        output_b = self.roberta2(input_ids=input_ids_b, attention_mask=attention_mask_b)
        output_c = self.roberta3(input_ids=input_ids_c, attention_mask=attention_mask_c)

        concatenated_output = torch.cat((output_a.pooler_output, output_b.pooler_output, output_c.pooler_output), 1)

        return self.classifier(concatenated_output)
    

class MultimodalRobertaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = response_columns
        self.labels = self.dataframe['Score'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        row = self.dataframe.iloc[index]
        response_a = row['Market.00a_OE']
        response_b = row['Market.00bc_OE']
        response_c = row['Market.00bc_OE follow up']
        score = row['Score']

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoding_a = self.tokenizer.encode_plus(response_a, add_special_tokens=True, max_length=self.max_length, padding='max_length', return_attention_mask=True, truncation=True)
        encoding_b = self.tokenizer.encode_plus(response_b, add_special_tokens=True, max_length=self.max_length, padding='max_length', return_attention_mask=True, truncation=True)
        encoding_c = self.tokenizer.encode_plus(response_c, add_special_tokens=True, max_length=self.max_length, padding='max_length', return_attention_mask=True, truncation=True)

        return {
            'input_ids_a': torch.tensor(encoding_a['input_ids'], dtype=torch.long),
            'attention_mask_a': torch.tensor(encoding_a['attention_mask'], dtype=torch.long),
            'input_ids_b': torch.tensor(encoding_b['input_ids'], dtype=torch.long),
            'attention_mask_b': torch.tensor(encoding_b['attention_mask'], dtype=torch.long),
            'input_ids_c': torch.tensor(encoding_c['input_ids'], dtype=torch.long),
            'attention_mask_c': torch.tensor(encoding_c['attention_mask'], dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float)
        }


# In[8]:


def split_and_load_dataset(df, batch_size=8, val=False):

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', output_attentions=False)
    if val == False:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Score'], random_state=42)
        
        train_dataset = MultimodalRobertaDataset(train_df, tokenizer)
        val_dataset = MultimodalRobertaDataset(val_df, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, None
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Score'], random_state=42)
        val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df['Score'], random_state=42)

        train_dataset = MultimodalRobertaDataset(train_df, tokenizer)
        val_dataset = MultimodalRobertaDataset(val_df, tokenizer)
        test_dataset = MultimodalRobertaDataset(test_df, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
train_loader, val_loader, test_loader = split_and_load_dataset(df, batch_size=2, val=False)


# In[9]:


def train(loss_fn, lr, EPOCH):
    model = MultimodalRoberta()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()
    for epoch in range(EPOCH):
        total_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch['input_ids_a'].to(device), batch['attention_mask_a'].to(device),
                batch['input_ids_b'].to(device), batch['attention_mask_b'].to(device),
                batch['input_ids_c'].to(device), batch['attention_mask_c'].to(device)
            )
            loss = loss_fn(outputs, batch['score'].to(device).long())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print("Average training loss: {0:.2f}".format(avg_train_loss))

    return model

MultimodalRobertaModel = train(nn.CrossEntropyLoss(), 1e-5, 5)


# In[10]:


def evaluate(model, val_loader):
    model.eval()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            outputs = model(
                batch['input_ids_a'].to(device), batch['attention_mask_a'].to(device),
                batch['input_ids_b'].to(device), batch['attention_mask_b'].to(device),
                batch['input_ids_c'].to(device), batch['attention_mask_c'].to(device)
            )
            #all_predictions.extend(outputs.cpu().numpy())
            #all_labels.extend(batch['score'].numpy())

            _, outputs = torch.max(outputs, 1)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(batch['score'].numpy())

    all_predictions = np.array(all_predictions).flatten()
    correct_predictions = sum(all_predictions == np.array(all_labels))
    total_predictions = len(all_predictions)
    test_accuracy = correct_predictions / total_predictions

    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

    return test_accuracy, all_predictions

test_accuracy, all_predictions = evaluate(MultimodalRobertaModel, val_loader)


# In[19]:


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Score'], random_state=42)
val_df['Predicted Score'] = all_predictions
val_df_inconsistent = val_df[val_df['Score'] != val_df['Predicted Score']]
## rank the order by index
val_df_inconsistent = val_df_inconsistent.sort_index()
val_df_inconsistent.to_excel(dataset_location + "val_df_inconsistent.xlsx")


# In[16]:


# draw the confusion matrix, do not use sns.heatmap, it will cause the error
## also show the rowsum and column sum on the plot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(val_df['Score'], val_df['Predicted Score'], labels=[0,1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1, 2, 3, 4])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix n = {len(val_df)}')
plt.show()


# In[13]:


difference_amount = val_df_inconsistent['Administration'].value_counts()
total = val_df['Administration'].value_counts()
difference_amount = difference_amount / total
difference_amount


# In[14]:


## Draw the histogram of val_df, with showing percentage of each score in each administration
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#sns.histplot(val_df['Score'], kde = False, bins = 5, discrete=True, ax=ax, label='Train')
# show the percentage of each administration in each score
sns.histplot(data=val_df, x='Score', hue='Administration', multiple='stack', ax=ax, bins=5, discrete=True)
plt.title('Score Distribution in Each Administration in test set')


# In[15]:


fix, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.histplot(data = train_df, x = 'Score', hue = 'Administration', multiple = 'stack', ax = ax, bins = 5, discrete = True)
plt.title('Score Distribution in Each Administration in train set')


# In[132]:


torch.cuda.empty_cache()


# In[45]:


learning_rates = [1e-5]  # List of learning rates to try
batch_sizes = [2, 4, 8, 16]  # List of batch sizes to try
epoch_sizes = [4,5,6,7,8,9,10]
combination_accuracies = {}


for epoch_size in epoch_sizes:
    for batch_size in batch_sizes:
        train_loader, val_loader, test_loader = split_and_load_dataset(df, batch_size=batch_size, val=False)
        model = train(nn.CrossEntropyLoss(), 1e-5, epoch_size)
        accuracy = evaluate(model, val_loader)
        combination_accuracies[(epoch_size, batch_size)] = accuracy
        torch.cuda.empty_cache()
        del model


# In[47]:


combination_accuracies


# In[ ]:




