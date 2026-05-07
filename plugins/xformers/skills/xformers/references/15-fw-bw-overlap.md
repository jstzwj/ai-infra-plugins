# 15 - Forward-Backward Overlap

## Overview

xFormers provides a forward-backward overlap mechanism inspired by DeepSeek's approach to overlapping communication and computation. This enables hiding communication costs behind computation in both the forward and backward passes.

**Source**: `xformers/fwbw_overlap.py`

## Key Idea

In distributed training with model/tensor parallelism:
1. Forward pass: compute → communicate → compute → communicate → ...
2. Backward pass: communicate → compute → communicate → compute → ...

By splitting the model into "chunks" (e.g., per transformer layer), we can overlap:
- Chunk N's backward pass computation with Chunk N+1's forward pass communication

This is done by running the backward pass in a background thread while the forward pass continues on the main thread.

## API Reference

### `overlap_fw_bw`

```python
from xformers import overlap_fw_bw

outputs = overlap_fw_bw(
    trigger_fw: Callable[[], T],       # Forward pass function
    trigger_bw: Callable[[], None],    # Backward pass function
    initial_bw_chunks: int = 0,        # Number of BW chunks to start early
) -> T                                  # Forward pass outputs
```

Runs forward and backward passes with overlap. The backward pass runs in a background thread.

**How it works:**
1. Starts a background autograd thread
2. Calls `before_forward(True)` to record FW chunks
3. Optionally starts some BW chunks early
4. Runs the forward pass
5. Triggers any remaining BW chunks
6. Waits for all BW chunks to complete

### `before_forward`

```python
from xformers.fwbw_overlap import before_forward

before_forward(
    record_fw_chunks: bool,  # Whether to record chunk boundaries
) -> None
```

Must be called before each forward pass. Resets the forward-backward state.

### `enter_comm`

```python
from xformers.fwbw_overlap import enter_comm

overlap_holder, *tensors = enter_comm(
    *tensors: torch.Tensor,
    name: str = "comm",
) -> tuple[EventOverlapHolder, Unpack[tuple[torch.Tensor, ...]]]
```

Marks the transition from compute to communication phase. Returns an `EventOverlapHolder` that should be passed to the corresponding `enter_compute`.

**In the backward pass:**
- The overlap holder waits for the communication to finish before computing

### `enter_compute`

```python
from xformers.fwbw_overlap import enter_compute

tensor = enter_compute(
    overlap_holder: EventOverlapHolder,
    *tensors: torch.Tensor,
    name: str = "compute",
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]
```

Marks the transition from communication to compute phase. Waits for the communication to complete.

**In the backward pass:**
- Creates a new overlap holder for the backward communication phase

## Core Classes

### `EventOverlapHolder`

A tensor subclass that holds a CUDA event for overlap synchronization:

```python
class EventOverlapHolder(torch.Tensor):
    event_overlap: Union[EventOverlap, None]
    _name: str

    @classmethod
    def capture(cls, device, name="") -> "EventOverlapHolder":
        """Create a holder with a new CUDA event"""

    def current_stream_wait(self):
        """Wait for the stored event on the current stream"""
```

### `PhaseBoundary`

Describes a boundary between communication and compute phases:

```python
@dataclass
class PhaseBoundary:
    fw_enter: str              # Name of the phase being entered
    arrived_sem: BoundedSemaphore
    unblock_sem: BoundedSemaphore
    fw_previous_boundary: Optional[PhaseBoundary]
    is_final: bool = True
```

### `_GlobalAutogradThread`

Background thread that runs backward passes:

```python
class _GlobalAutogradThread:
    thread: Optional[Thread] = None
    todo: SimpleQueue  # Queue of (backward_fn, release_semaphore)

    @classmethod
    def run(cls):
        """Main loop: dequeue and run backward functions"""

    @classmethod
    def cleanup_at_exit(cls):
        """Shutdown the background thread"""
```

## Usage Pattern

### Basic Overlap Pattern

```python
import torch
from xformers.fwbw_overlap import (
    overlap_fw_bw,
    before_forward,
    enter_comm,
    enter_compute,
)

class OverlappingModel(nn.Module):
    def __init__(self, layers, process_group):
        self.layers = layers
        self.pg = process_group

    def forward(self, x):
        for layer in self.layers:
            # Enter communication (all-reduce, etc.)
            overlap, x = enter_comm(x, name=f"comm_{i}")

            # Enter computation (attention, MLP, etc.)
            x = enter_compute(overlap, x, name=f"compute_{i}")

            # Actual layer computation
            x = layer(x)

        return x

    def trigger_backward(self):
        """Triggers backward pass in background"""
        loss = self.output.sum()
        loss.backward()

# Training loop
def train_step(model, x):
    def fw():
        before_forward(True)
        model.output = model(x)
        return model.output

    def bw():
        model.trigger_backward()

    output = overlap_fw_bw(fw, bw, initial_bw_chunks=0)
    return output
```

### With DeepEP

When DeepEP is available, `EventOverlap` provides true CUDA event overlap:

```python
try:
    from deep_ep.utils import EventHandle, EventOverlap
except ImportError:
    # Fallback: standard CUDA events
    class EventHandle:
        def __init__(self):
            self._event = torch.cuda.Event()
        def current_stream_wait(self):
            self._event.wait()
```

## Execution Order

The overlap mechanism ensures this execution order:

```
Forward:
  [compute_0] → [comm_0] → [compute_1] → [comm_1] → ...
                     ↓            ↓              ↓
Backward:           ...    [bw_chunk_0]   [bw_chunk_1]  ...
```

Where each `bw_chunk_i` overlaps with the next forward phase's communication.

## Internal State

### `_CurrentForwardState`

Global state tracking the current forward-backward overlap:

```python
class _CurrentForwardState:
    record_fw_chunks: ClassVar[bool] = False
    bw_chunks_wait: ClassVar[bool] = False
    fw_previous_boundary: ClassVar[Optional[PhaseBoundary]] = None
    bw_last_boundary: ClassVar[Optional[Union[PhaseBoundary, Callable]]] = None
    bw_done_semaphore: ClassVar[Optional[Semaphore]] = None
    bw_exception: ClassVar[Optional[Exception]] = None
    cv_on_bw: ClassVar[Condition] = threading.Condition()
```

### Autograd Functions

Three custom autograd functions handle the overlap:

1. **`_ExitCompute`**: In forward, captures a CUDA event. In backward, waits for communication.
2. **`_EnterCompute`**: In forward, waits for communication. In backward, captures a CUDA event.
3. **`_WaitInBW`**: In backward, signals that a boundary was reached and waits for the unblock signal.

## Safety Features

1. **Timeout**: All semaphore acquisitions have a 3800-second timeout
2. **Exception propagation**: Backward exceptions are caught and re-raised in the forward thread
3. **At-exit cleanup**: Background thread is properly shut down on process exit
4. **Single BW pass check**: `_FillGradientForOverlapHolder` detects multiple BW passes (would indicate a bug)

## Limitations

1. **Single GPU**: If there's only 1 GPU, `EventOverlapHolder` still works but doesn't provide overlap benefit
2. **Thread safety**: The background thread runs autograd in a separate Python thread
3. **CUDA events**: Relies on CUDA event ordering for correctness
4. **Debug complexity**: Overlapping forward and backward passes makes debugging more difficult
