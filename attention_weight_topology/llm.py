from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tda import Graph
import torch

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# input_text = "a b c d e"
input_text = "The capital of Canada is Ottawa. The capital of the United States is Washington, D.C. The capital of France is"

names = list(input_text.split())

def get_attention_weights(model, input_text):
    """
    For this to return attention weights, we make a modification to GPT2 to use eager attention. This was done in the code. We cannot return weights with the native SDPA implementation.
        transformers/models/gpt2/modeling_gpt2.py:987     config._attn_implementation = "eager"
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.forward(**inputs, output_attentions=True)
    # generated_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.7, do_sample=True)
    # model_generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return torch.stack(outputs.attentions).squeeze(1)

def get_graph(
        token_indices: tuple[int], layer: int, head: int, 
        attention_weights: torch.Tensor,
        names: list[str] = None
    ) -> Graph:
    """
    Get a graph from attention weights for specified token indices, layer, and head.
    Args:
        token_indices (tuple[int]): [start, end)
        layer (int): Layer index
        head (int): Head index
        attention_weights (torch.Tensor): Attention weights tensor of shape (num_layers, num_heads, seq_len, seq_len)
    """
    start, end = token_indices
    attn_matrix = attention_weights[layer, head][start:end, start:end]
    mask = attn_matrix > 0.0
    adj = attn_matrix * mask.float()
    return Graph(adj_matrix=adj, names=names)


A = get_attention_weights(model, input_text)
A = A.detach().cpu()

G = get_graph((0, 4), layer=0, head=0, attention_weights=A)

# Next we build a layer-wise x token-wise procession (list of graphs)
procession = [G]

start_token_idx = 1
head = 0
seq_len = A.shape[2]
n_layers = A.shape[0]

for layer in range(n_layers):
    for token_idx in range(start_token_idx, seq_len):
        G_ = get_graph((0, token_idx + 1), layer=layer, head=head, attention_weights=A)
        G_.build_connected_components()
        procession.append(G_)




