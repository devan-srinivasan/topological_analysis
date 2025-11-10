import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from analysis import rips

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# input_text = "a b c d e"
input_text = "The capital of Canada is Ottawa. The capital of the United States is Washington, D.C. The capital of France is"

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.forward(**inputs, output_hidden_states=True)
hs = torch.stack(outputs.hidden_states, dim=1)

distance_matrix = torch.cdist(hs[0, 0], hs[0, 0], p=2)
complex = rips(
    hs[0, 0].detach().numpy(),
    max_edge_length=distance_matrix.max().item(),
    max_dimension=4,
)
print(complex.betti_numbers())