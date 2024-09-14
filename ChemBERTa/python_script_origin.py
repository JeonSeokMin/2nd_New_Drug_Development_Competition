from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the ChemBERTa model and tokenizer
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Load data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Tokenize SMILES strings with dynamic padding
def tokenize_smiles(smiles):
    return tokenizer(smiles, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Apply tokenization and store as a list
train_tokens = [tokenize_smiles(smiles) for smiles in train_df['Smiles']]
test_tokens = [tokenize_smiles(smiles) for smiles in test_df['Smiles']]

# No scaling applied to IC50 values
y = train_df['IC50_nM'].values

# Split into training and validation sets
train_tokens, val_tokens, y_train, y_val = train_test_split(train_tokens, y, test_size=0.2, random_state=42)

class SMILESDataset(Dataset):
    def __init__(self, tokens, targets=None):
        self.tokens = tokens
        self.targets = targets
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.tokens[idx].items()}
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

train_dataset = SMILESDataset(train_tokens, y_train)
val_dataset = SMILESDataset(val_tokens, y_val)
test_dataset = SMILESDataset(test_tokens)

# Use DataCollatorWithPadding to handle dynamic padding
data_collator = DataCollatorWithPadding(tokenizer)

# Prepare training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Predict on the test set
predictions = trainer.predict(test_dataset)

# Save the predictions to a CSV file
output_df = pd.DataFrame({'ID': test_df['ID'], 'IC50_nM': predictions.predictions.squeeze()})
output_df.to_csv('../data/results/predictions_chembert_v1.csv', index=False) # 0.6297