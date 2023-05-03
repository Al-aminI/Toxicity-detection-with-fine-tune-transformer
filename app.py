from flask import Flask, render_template, session, url_for, Response, request, flash, redirect, jsonify
import requests
from torchtext.data.utils import get_tokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import torch.nn as nn

model = GPT2Model.from_pretrained("gpt2")
embedding_layer = model.get_input_embeddings()

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim)
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(x[:, -1, :])
        return x



model_state_dict = torch.load("toxicity model/toxicity_model.pt")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")




app = Flask(__name__)
app.app_context().push()


@app.route('/<text>')
def toxic_det(text):
    with torch.no_grad():
        inpt = "you dey craze"
        input_ids = tokenizer.encode(inpt, return_tensors="pt")
        output = model_state_dict(input_ids)
        out = output.argmax(1)
        if out == 0:
            return {"msg":"Action is needed"}
        else:
            return {"msg":"no action needed"}
              
    

if __name__ == '__main__':
    app.run(debug=True)

