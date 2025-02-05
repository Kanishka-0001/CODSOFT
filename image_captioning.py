import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM
from PIL import Image
import numpy as np
import nltk
import matplotlib.pyplot as plt

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model.eval()
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(image)
    features = features.view(features.size(0), -1)
    return features

class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs

nltk.download('punkt')

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer, max_length=20):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        image_features = extract_image_features(image_path)
        tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        caption_ids = tokens['input_ids'].squeeze(0)
        return image_features, caption_ids

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_paths = ['image1.jpg', 'image2.jpg']
captions = ['a dog playing with a ball', 'a man riding a bike']

dataset = ImageCaptionDataset(image_paths, captions, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

embed_size = 256
hidden_size = 512
vocab_size = len(tokenizer.vocab)
model = CaptionGenerator(embed_size, hidden_size, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for image_features, caption_ids in dataloader:
        optimizer.zero_grad()
        outputs = model(image_features, caption_ids[:, :-1])
        loss = criterion(outputs.view(-1, vocab_size), caption_ids[:, 1:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

def generate_caption(image_path, model, tokenizer, max_length=20):
    model.eval()
    image_features = extract_image_features(image_path)
    caption = ['<start>']
    
    for _ in range(max_length):
        caption_ids = tokenizer.encode(' '.join(caption), return_tensors='pt')
        outputs = model(image_features, caption_ids)
        predicted_id = outputs.argmax(dim=-1)[:, -1].item()
        predicted_word = tokenizer.decode(predicted_id)
        caption.append(predicted_word)
        
        if predicted_word == '<end>':
            break
    
    return ' '.join(caption[1:])

generated_caption = generate_caption('new_image.jpg', model, tokenizer)
print(f"Generated Caption: {generated_caption}")

