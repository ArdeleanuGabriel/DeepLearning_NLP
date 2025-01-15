import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class StockNewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_window(data, sequence_length, feature_columns, target_col):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        window = data.iloc[i:i + sequence_length][feature_columns].values
        X.append(window)
        price_change = data.iloc[i + sequence_length][target_col]
        if price_change == 1:
            y.append([1, 0])  # Increase
        else:
            y.append([0, 1])  # Decrease
    return np.array(X), np.array(y)


def prepare_data(stock_data, news_data, sequence_length=5):
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    news_data["date"] = pd.to_datetime(news_data["date"])

    data = pd.merge(stock_data, news_data, on="date", how="left")
    data.fillna({"text_sentiment": 0, "key_words": "", "title": ""}, inplace=True)

    scaler = StandardScaler()
    data["text_sentiment"] = scaler.fit_transform(data[["text_sentiment"]])

    vectorizer = CountVectorizer(max_features=100)
    key_words_encoded = vectorizer.fit_transform(data["key_words"]).toarray()

    key_words_columns = [f"kw_{i}" for i in range(key_words_encoded.shape[1])]
    key_words_df = pd.DataFrame(key_words_encoded, columns=key_words_columns, index=data.index)
    data = pd.concat([data, key_words_df], axis=1)

    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["price_change"] = (data["close"] > data["open"]).astype(int)

    feature_columns = ["text_sentiment"] + key_words_columns

    X, y = create_sliding_window(data, sequence_length, feature_columns, "price_change")
    return X, y


def train_model(model, dataloader, criterion, optimizer, num_epochs, device, model_name):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Get the class labels from one-hot encoded targets
            y_batch_indices = torch.argmax(y_batch, dim=1)

            # Get the model's outputs (probabilities)
            outputs = model(X_batch)

            # Compute the loss
            loss = criterion(outputs, y_batch_indices)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compare probabilities of the two classes (increase vs decrease)
            increase_prob = outputs[:, 0]  # Probability of increase (class 0)
            decrease_prob = outputs[:, 1]  # Probability of decrease (class 1)
            predictions = (increase_prob > decrease_prob).long()  # Predict class 0 if increase prob > decrease prob, else class 1

            # Update correct and total counts
            correct += (predictions == y_batch_indices).sum().item()
            total += y_batch_indices.size(0)
            epoch_loss += loss.item()

        # Calculate and print accuracy
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save model weights after training
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
            outputs = model(X_batch)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


class StockPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPricePredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm6 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm7 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm8 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm9 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm10 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1000)
        self.fc2 = nn.Linear(1000, 10000)
        self.fc3 = nn.Linear(10000, 10000)
        self.fc4 = nn.Linear(10000, 500)
        self.fc5 = nn.Linear(500, 100)
        self.fc6 = nn.Linear(100, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out4, _ = self.lstm4(lstm_out3)
        lstm_out5, _ = self.lstm5(lstm_out4)
        lstm_out6, _ = self.lstm6(lstm_out5)
        lstm_out7, _ = self.lstm7(lstm_out6)
        lstm_out8, _ = self.lstm8(lstm_out7)
        lstm_out9, _ = self.lstm9(lstm_out8)
        lstm_out10, _ = self.lstm10(lstm_out9)

        last_output = lstm_out10[:, -1, :]

        fc_out1 = torch.relu(self.fc1(last_output))
        fc_out2 = torch.relu(self.fc2(fc_out1))
        fc_out3 = torch.relu(self.fc3(fc_out2))
        fc_out4 = torch.relu(self.fc4(fc_out3))
        fc_out5 = torch.relu(self.fc5(fc_out4))
        fc_out6 = self.fc6(fc_out5)

        output = self.softmax(fc_out6)
        return output


if __name__ == "__main__":
    stock_folder = "SelectedHistoricalFolder"
    news_file = "df_0_news_with_keywords_and_sentiment.parquet"

    all_stock_data = []
    for file in os.listdir(stock_folder):
        if file.endswith(".csv"):
            stock_file_path = os.path.join(stock_folder, file)
            stock_data = pd.read_csv(stock_file_path)
            stock_data["date"] = pd.to_datetime(stock_data["date"])
            all_stock_data = [stock_data]

            stock_data = pd.concat(all_stock_data, ignore_index=True)
            news_data = pd.read_parquet(news_file)
            news_data["date"] = news_data["time"].dt.date

            sequence_length = 5
            X, y = prepare_data(stock_data, news_data, sequence_length)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            train_dataset = StockNewsDataset(X_train, y_train)
            test_dataset = StockNewsDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            input_dim = X_train.shape[2]
            hidden_dim = 512
            output_dim = 2
            num_layers = 5
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = StockPricePredictor(input_dim, hidden_dim, output_dim, num_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 10
            train_model(model, train_loader, criterion, optimizer, num_epochs, device, file.replace('.csv', '')+"_2_model_weights")

            evaluate_model(model, test_loader, device)
