import torch
import torch.nn as nn
import torch.optim as optim
import json
from transformers import BertTokenizer
import torch
import pandas as pd
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
unequal_df = pd.read_csv('Data/cleaned_non_equal.csv')
class ANNModel(nn.Module):
  def __init__(self):
    super(ANNModel, self).__init__()

    # Define layers
    self.fc1 = nn.Linear(2048, 1024)  # First hidden layer
    self.dropout1 = nn.Dropout(p=0.2)  # Dropout layer after first hidden layer (20% dropout)
    self.fc2 = nn.Linear(1024, 512)  # Second hidden layer
    self.dropout2 = nn.Dropout(p=0.2)  # Dropout layer after second hidden layer (20% dropout)
    self.fc3 = nn.Linear(512, 5)  # Output layer with 5 units

  def forward(self, x):
    # Forward pass through the network
    x = self.fc1(x)
    x = self.dropout1(x)
    x = nn.functional.relu(x)  # Apply ReLU activation after first hidden layer
    x = self.fc2(x)
    x = self.dropout2(x)
    x = nn.functional.relu(x)  # Apply ReLU activation after second hidden layer
    x = self.fc3(x)
    x = torch.sigmoid(x)  # Apply sigmoid activation for binary outputs
    return x


# Create an instance of the model
model = ANNModel()

# Define loss function (Binary Cross Entropy Loss) and optimizer (Adam)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__=="__main__":
    # Data preparation
    x_indexes = json.loads(open('scoring_responses\prompts\indexes.json','r').read())
    X_train = []
    y_train = json.loads(open('training_data/y_train.json','r').read())[:66]
    for i in x_indexes:
       context_tokens = tokenizer.encode(unequal_df['prev_context'][i], add_special_tokens=True, max_length=1024, pad_to_max_length=True, truncation=True)
       response_token = tokenizer.encode(unequal_df['response'][i], add_special_tokens=True, max_length=512, pad_to_max_length=True, truncation=True)
       agent_response_token = tokenizer.encode(unequal_df['agent_response'][i], add_special_tokens=True, max_length=512, pad_to_max_length=True, truncation=True)
       _ = context_tokens
       _.extend(response_token)
       _.extend(agent_response_token)
       X_train.append(_)
    len(X_train)
    y_train = torch.tensor(y_train).float()
    json.dump(X_train,open('training_data/X_train.json','w+'))
    X_train = torch.tensor(X_train).float()
    epochs = 50
    # Training loop
    for epoch in range(epochs):
        # Set model to train mode
        model.train()
        # Forward pass
        outputs = model(X_train)
        # Compute loss
        loss = criterion(outputs, y_train)
        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss every few epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print('Training finished.')
    torch.save(model.state_dict(), 'model/torch_ann_model.pth')
