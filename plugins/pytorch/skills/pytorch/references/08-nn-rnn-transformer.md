# PyTorch - Chapter 8: Recurrent and Transformer Layers

This reference covers RNN, LSTM, GRU, their cell variants, PackedSequence, and the complete Transformer architecture.

---

## 8.1 Recurrent Neural Networks

### nn.RNN

```python
nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh',
       bias=True, batch_first=False, dropout=0.0, bidirectional=False,
       device=None, dtype=None)
```

- **input_size**: Features in input x
- **hidden_size**: Features in hidden state h
- **num_layers**: Stacked RNN layers
- **nonlinearity**: 'tanh' or 'relu'
- **batch_first**: If True, input shape is (batch, seq, feature)
- **dropout**: Dropout probability between layers (not on last layer)
- **bidirectional**: If True, bidirectional RNN

**Input**: `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` if batch_first
**Output**: `(output, h_n)`
- **output**: `(seq_len, batch, D*hidden_size)` where D=2 if bidirectional
- **h_n**: `(D*num_layers, batch, hidden_size)`

```python
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
input = torch.randn(32, 5, 10)   # batch=32, seq_len=5, features=10
h0 = torch.randn(2, 32, 20)     # D*num_layers=2, batch=32, hidden=20
output, hn = rnn(input, h0)
# output: (32, 5, 20), hn: (2, 32, 20)
```

### nn.LSTM

```python
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        batch_first=False, dropout=0.0, bidirectional=False,
        proj_size=0, device=None, dtype=None)
```

Same parameters as RNN plus:
- **proj_size**: If > 0, LSTM with projected outputs (reduces hidden state size)

**Input**: `(seq_len, batch, input_size)`
**Output**: `(output, (h_n, c_n))`
- **output**: `(seq_len, batch, D*proj_size)` or `(seq_len, batch, D*hidden_size)`
- **h_n**: `(D*num_layers, batch, proj_size)` or `(D*num_layers, batch, hidden_size)`
- **c_n**: `(D*num_layers, batch, hidden_size)`

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
input = torch.randn(32, 5, 10)
h0 = torch.randn(2, 32, 20)
c0 = torch.randn(2, 32, 20)
output, (hn, cn) = lstm(input, (h0, c0))
# output: (32, 5, 20)
```

### nn.GRU

```python
nn.GRU(input_size, hidden_size, num_layers=1, bias=True,
       batch_first=False, dropout=0.0, bidirectional=False,
       device=None, dtype=None)
```

```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
input = torch.randn(32, 5, 10)
output, hn = gru(input)
```

---

## 8.2 RNN Cell Variants

### nn.RNNCell

```python
nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
```

Single timestep: **h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)**

```python
cell = nn.RNNCell(10, 20)
input = torch.randn(32, 10)
hx = torch.randn(32, 20)
hx = cell(input, hx)  # One timestep
```

### nn.LSTMCell

```python
nn.LSTMCell(input_size, hidden_size, bias=True, device=None, dtype=None)
```

```python
cell = nn.LSTMCell(10, 20)
input = torch.randn(32, 10)
hx = torch.randn(32, 20)
cx = torch.randn(32, 20)
hx, cx = cell(input, (hx, cx))
```

### nn.GRUCell

```python
nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
```

---

## 8.3 PackedSequence

For variable-length sequences in RNNs:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pad sequences to equal length
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
lengths = torch.tensor([3, 2, 1])
padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
# tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])

# Pack (more efficient for RNNs)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
output_packed, hidden = lstm(packed)

# Unpack
output, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
```

---

## 8.4 Transformer

### nn.Transformer

```python
nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
               dim_feedforward=2048, dropout=0.1, activation='relu',
               custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5,
               batch_first=False, norm_first=False, bias=True,
               device=None, dtype=None)
```

```python
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                              batch_first=True)
src = torch.randn(32, 10, 512)   # batch=32, seq=10, d_model=512
tgt = torch.randn(32, 20, 512)
output = transformer(src, tgt)   # (32, 20, 512)
```

### nn.TransformerEncoder

```python
nn.TransformerEncoder(encoder_layer, num_layers, norm=None,
                       enable_nested_tensor=True, mask_check=True)
```

### nn.TransformerEncoderLayer

```python
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                           activation='relu', layer_norm_eps=1e-5,
                           batch_first=False, norm_first=False, bias=True,
                           device=None, dtype=None)
```

### nn.TransformerDecoder

```python
nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

### nn.TransformerDecoderLayer

```python
nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                           activation='relu', layer_norm_eps=1e-5,
                           batch_first=False, norm_first=False, bias=True,
                           device=None, dtype=None)
```

### nn.MultiheadAttention

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True,
                      add_bias_kv=False, add_zero_attn=False,
                      kdim=None, vdim=None, batch_first=False,
                      device=None, dtype=None)
```

```python
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
query = torch.randn(32, 10, 512)
key = torch.randn(32, 20, 512)
value = torch.randn(32, 20, 512)
attn_output, attn_weights = mha(query, key, value)
# attn_output: (32, 10, 512), attn_weights: (32, 10, 20)
```

### F.scaled_dot_product_attention

```python
torch.nn.functional.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
    scale=None, enable_gqa=False
)
```

Supports Flash Attention, memory-efficient attention, and math fallback automatically.
