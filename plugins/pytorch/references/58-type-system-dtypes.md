# PyTorch - Chapter 58: Type System and dtypes

This reference covers PyTorch's type system, dtype promotion, and symbolic shapes.

---

## 58.1 Complete dtype Reference

| dtype | Bytes | Alias | Range |
|-------|-------|-------|-------|
| torch.float16 | 2 | half | ±65504 |
| torch.float32 | 4 | float | ±3.4e38 |
| torch.float64 | 8 | double | ±1.8e308 |
| torch.bfloat16 | 2 | - | Same range as float32, less precision |
| torch.int8 | 1 | - | -128 to 127 |
| torch.int16 | 2 | short | -32768 to 32767 |
| torch.int32 | 4 | int | ±2.1e9 |
| torch.int64 | 8 | long | ±9.2e18 |
| torch.uint8 | 1 | byte | 0 to 255 |
| torch.bool | 1 | - | True/False |
| torch.complex64 | 8 | cfloat | Two float32 |
| torch.complex128 | 16 | cdouble | Two float64 |

---

## 58.2 Type Promotion Rules

1. Same type → no promotion
2. Complex + Real → Complex (wider float)
3. Float + Int → Float
4. Bool promotes to any type
5. Between integers → wider type

---

## 58.3 SymInt / SymFloat / SymBool

```python
# Symbolic integers for dynamic shapes in torch.compile
s = torch.SymInt(5)
s + 3        # SymInt(8)

torch.sym_int(x)
torch.sym_float(x)
torch.sym_max(a, b)
torch.sym_min(a, b)
torch.sym_ite(pred, a, b)
torch.sym_not(pred)
torch.sym_sum([a, b, c])
```

---

## 58.4 Casting Methods

```python
t.float()      # → float32
t.double()     # → float64
t.half()       # → float16
t.bfloat16()   # → bfloat16
t.int()        # → int32
t.long()       # → int64
t.bool()       # → bool
t.to(dtype)    # → specified dtype
```
