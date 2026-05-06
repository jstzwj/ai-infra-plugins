# Programming Model

Tile IR extends CUDA's low-level programming model with new abstractions that differ from what has previously existed in CUDA C++ or PTX. This section introduces the programming model of Tile IR and familiarizes the reader with its core concepts and abstractions through a series of real programs, building up to a dynamic, high-performance implementation of GEMM.

## Tile Kernels

Tile IR programs are referred to as **tile kernels**, which like CUDA C++ or PTX, are functions which run as N copies in parallel when invoked. The primary difference is the basic unit of execution: a **tile-block**, which expresses the computation performed by a single logical tile thread operating over a multi-dimensional tile of data.

During execution, each tile kernel is referred to as a tile kernel instance. Below is a simple Tile IR kernel which prints "Hello World!":

```cuda_tile
cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}
```

### What's Different About Tile Programming?

Tile IR is an extension to the CUDA programming model that enables first class support for tile programming. Tiled kernels express programs as a grid of logical tile threads that operate over tiles. The mapping of both the grid and individual tile threads to the underlying hardware's threads is abstracted away from the programming model and is handled by the compiler.

The SIMT programming model of NVIDIA's streaming multiprocessor (SM) is one in which threads operate over (relatively) small pieces of data and the user is responsible for dividing and scheduling the threads into the appropriate blocks to compute over the input data in an efficient manner. This model gives flexibility to programmers on how to map threads to data, or vice-versa. SIMT is the programming model exposed by CUDA and PTX and has served NVIDIA GPUs well since its introduction in 2006.

The rise in importance of deep learning has both introduced a greater regularity to user workloads and an ever increasing need to deliver performance for these workloads. This has led to new specialized hardware in the form of tensor cores.

Tensor cores introduce a new dimension to the SIMT programming model. Now, SM threads must cooperate with the tensor cores in order to reach peak performance. With each new generation of hardware the interplay between these two pieces of silicon has unlocked amazing new performance but with increasing programming complexity.

Tile IR has been built to aid in the implementation of high-performance algorithms that take full advantage of the underlying hardware's capabilities while mitigating the increase in programming complexity.

By abstracting thread-to-data mapping, Tile IR simplifies the use of specialized hardware like Tensor Cores compared to traditional SIMT models.

### Kernel I/O

To illustrate the design of Tile IR we will move from our simple hello world kernel to one which implements 1-d tensor (i.e., vector) addition with a fixed block size 128.

Tile kernels accept inputs and outputs as parameters; this is the only mechanism for consuming and producing data, so we start by defining the kernel parameters.

```cuda_tile
entry @vector_block_add_128x1_kernel(
    %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)
```

The above code fragment defines a kernel named `vector_block_add_128x1_kernel` which takes three arguments, representing the two input buffers a and b, and the output buffer c. The types of all the arguments are scalar pointers, which are represented as zero dimensional tensors containing a single pointer.

All values in Tile IR are either tensors, or tensor views. A tensor is an n-dimensional rectangular array, described by its rank (number of dimensions), the shape (extent along each dimension), and its primitive element type. Tensors may have a rank of 0 or higher. Rank-0 tensors are scalars. The rank, dimensions, and element type are all part of the tensor type and are statically known. Tensor types are assigned to values which represent a logical view of a multidimensional array contained in global memory. Global memory is always accessed via tensors, and thus pointer arguments always point to CUDA device allocations in global device memory. Tile kernels do not have return values and thus omit a return type annotation.

A common pattern in the Tile IR programming model is for kernels to take unstructured base pointers as parameters which can then be used to construct the required tensor. This flexibility gives rise to multiple ways to use pointers depending on the desired program behavior.

We must take a few steps to convert a base pointer `%a_ptr_base_scalar` into a tensor of arbitrary pointers representing the 128x1 tile we want to compute on.

**Step 1:** Create an offset tensor which represents the inclusive (0, 127) interval. We use `cuda_tile.iota` which constructs a range tensor that counts from 0 to n - 1:

```cuda_tile
%offset = iota : tile<128xi32>
```

**Step 2:** Reshape from a scalar `ptr<f32>` to a 1-d tensor `1xptr<f32>` so we have the correct rank:

```cuda_tile
%a_ptr_base_tensor = reshape %a_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1xptr<f32>>
```

**Step 3:** Broadcast the pointer so we have a 1-d tensor of (base, ..., base) containing 128 elements:

```cuda_tile
%a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
```

**Step 4:** Add the offset tensor to the tensor of pointers:

```cuda_tile
%a_tensor = offset %a_ptr, %offset :
    tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
```

**Step 5:** Load both operands, perform the addition, and store to the output:

```cuda_tile
%a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
```

We now have a complete kernel for a single tile-block that performs addition over 128 element vectors. As you can see this code is written from a single thread of control, but its level of parallelism will be determined by the compiler.

## Tile Grid

So far we have only examined kernels which are written for a single tile block. Tile IR allows tile blocks to be grouped into a **tile grid**, similar to CUDA C++, enabling users to launch sets of tile blocks that execute in parallel. Tile kernels, as with PTX, are implicitly parameterized over the tile block coordinates (which can be queried via `cuda_tile.get_tile_block_id`) and can be 1-d, 2-d, or 3-d.

When a tile kernel is launched the user specifies the grid size, which determines the number of tile blocks launched. The number of tile blocks launched is equal to the size of the grid. For example if we launch our previous example with a (1, 1, 2) grid, we will run an identical computation twice.

Here is an improved hello world program which shows off querying the grid size and coordinates:

```cuda_tile
cuda_tile.module @hello_world_module {
    // TileIR kernel function
    entry @hello_world_kernel() {
        // Step 1. Get the tile block ID
        %block_x_index, %block_y_index, %block_z_index = cuda_tile.get_tile_block_id : tile<i32>

        // Step 2. Get the tile block dimensions
        %block_dim_x, %block_dim_y, %block_dim_z = cuda_tile.get_num_tile_blocks : tile<i32>

        // Step 3. Print the tile block ID and dimensions. Each tile executes the
        // following print statement and prints a single line.
        cuda_tile.print "Hello, I am tile <%i, %i, %i> in a kernel with <%i, %i, %i> tiles.\n",
            %block_x_index, %block_y_index, %block_z_index, %block_dim_x, %block_dim_y, %block_dim_z
            : tile<i32>, tile<i32>, tile<i32>,
              tile<i32>, tile<i32>, tile<i32>
        }
}
```

If we use a grid that is (1, 1, 2) we will see two prints:

```
      "Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 2> tiles."
      "Hello, I am tile <0, 0, 1> in a kernel with <1, 1, 2> tiles."
```

## Implementing GEMM

Now that we understand the basics concepts of the Tile IR programming model, we will introduce how to compute a 2-d GEMM for a single block, and then generalize it step by step to a full GEMM by utilizing the tile grid and introducing control flow and manual tiling.

### GEMM with a Single Block

Let's first start by naturally moving from a single static block vector addition to a single static square block matrix multiplication.

```cuda_tile
cuda_tile.module @gemm_block_64x64_module {
    entry @gemm_block_64x64_kernel(
        %a_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) {

    %offset_flat = iota : tile<4096xi32>
    %offset = reshape %offset_flat :
        tile<4096xi32> -> tile<64x64xi32>

    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %a_tensor = offset %a_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // Now we do the same for B.
    %b_ptr_base_tensor = reshape %b_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %b_ptr = broadcast %b_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %b_tensor = offset %b_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // And the same for C.
    %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %c_ptr = broadcast %c_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %c_tensor = offset %c_ptr, %offset :
         tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // Load a single 64x64 matrix from the tile.
    %A_block, %token_a = load_ptr_tko weak %a_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    // Load a single 64x64 matrix from the tile.
    %B_block, %token_b = load_ptr_tko weak %b_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    %init_accum = cuda_tile.constant <f32: 0.000000e+00> : !cuda_tile.tile<64x64xf32>

    %C_block = mmaf %A_block, %B_block, %init_accum: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

    store_ptr_tko weak %c_tensor, %C_block :
        tile<64x64xptr<f32>>, tile<64x64xf32> -> token
    }
}
```

### GEMM Block by Block

Let us look at how to generalize this to work for a large matrix size -- say 4096x4096, where each tile block will compute a single output tile of the final matrix.

```cuda_tile
cuda_tile.module @gemm_square_4096_tile_64x64_module {
    entry @gemm_square_4096_tile_64x64_kernel(
        %a_ptr_base_scalar: tile<ptr<f32>>,
        %b_ptr_base_scalar: tile<ptr<f32>>,
        %c_ptr_base_scalar: tile<ptr<f32>>
    ) {
        // Read Tile block id's.
        %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

        // We assume we have tiled a 4096x4096 @ 4096x4096 matrix split into
        // 64x64 tiles so the tile m, n, k are all 64.
        %m_tile_size = constant <i32: 64> : tile<i32>
        %m_stride_factor = cuda_tile.constant <i32: 64> : tile<64x64xi32>
        %k_tile_size = cuda_tile.constant <i32: 64> : tile<i32>

        %range_start = cuda_tile.constant <i32: 0> : tile<i32>
        %range_step = cuda_tile.constant <i32: 1> : tile<i32>
        %init_accum = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>

        // The shared range from (0, 63).
        %tile_size_range = cuda_tile.iota : tile<64xi32>
```

We must first compute the tensor of offsets for A and B so that we can obtain a tensor of pointers for each in order to load them from memory.

Conceptually we start by computing the offsets of the M dimension:

```python
m_offsets = block_x_index * k_tile_size + arange(0, k_tile_size)
```

This produces a vector starting from the "top-corner" of the tile at `(block_x_index * k_tile_size, block_x_index * k_tile_size + k_tile_size)`.

```cuda_tile
        %a_tile_base = cuda_tile.muli %block_x_index, %k_tile_size : tile<i32>
        %a_tile_base_reshape = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
        %a_tile_base_tensor = cuda_tile.broadcast %a_tile_base_reshape :
            tile<1xi32> -> tile<64xi32>
        %m_offsets_vec = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<64xi32>

        // Broadcast the m_offsets into a matrix where each column is identical and scaled by stride.
        %m_offsets_matrix = cuda_tile.reshape %m_offsets_vec :
            tile<64xi32> -> tile<64x1xi32>
        %m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix :
            tile<64x1xi32> -> tile<64x64xi32>
        %m_offsets = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        // Broadcast the k_offsets into a matrix where row is identical and scaled by stride.
        %ak_offsets_matrix = cuda_tile.reshape %tile_size_range :
             tile<64xi32> -> tile<1x64xi32>
        %ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix :
            tile<1x64xi32> -> tile<64x64xi32>
        %ak_offsets = cuda_tile.muli %ak_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        // Finally we add them together resulting in the final offset matrix for A.
        %a_tile_offsets = cuda_tile.addi %m_offsets, %ak_offsets : tile<64x64xi32>
```

The resulting offset matrix for A would look like:

```
   [[   0,    1,    2,  ...,   61,   62,   63],
    [  64,   65,   66,  ...,  125,  126,  127],
    [ 128,  129,  130,  ...,  189,  190,  191],
       ...,
    [3904, 3905, 3906,  ..., 3965, 3966, 3967],
    [3968, 3969, 3970,  ..., 4029, 4030, 4031],
    [4032, 4033, 4034,  ..., 4093, 4094, 4095]]
```

Now we do the same for B:

```cuda_tile
        %b_tile_base = cuda_tile.muli %block_y_index, %k_tile_size : tile<i32>
        %b_tile_base_reshape = cuda_tile.reshape %b_tile_base :
            tile<i32> -> tile<1xi32>
        %b_tile_base_tensor = cuda_tile.broadcast %b_tile_base_reshape :
            tile<1xi32> -> tile<64xi32>
        %n_offsets_vec = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<64xi32>
        %bk_offsets_matrix = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
        %bk_offsets = cuda_tile.broadcast %bk_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
        %n_offsets_matrix = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
        %n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix :  tile<1x64xi32> -> tile<64x64xi32>
        %n_offsets = cuda_tile.muli %n_offsets_broadcast, %m_stride_factor : tile<64x64xi32>
        %b_tile_offsets = cuda_tile.muli %bk_offsets, %n_offsets : tile<64x64xi32>
```

Now that we have computed the initial offsets for the pointers we convert the base pointers and add the offsets:

```cuda_tile
        %a_ptr_base_tensor = cuda_tile.reshape %a_ptr_base_scalar :
            tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %a_ptr = cuda_tile.broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
        %a_tile_ptr = offset %a_ptr, %a_tile_offsets :
            tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

        // And the same for B.
        %b_ptr_tile_tensor = reshape %b_ptr_base_scalar :
            tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %b_ptr = broadcast %b_ptr_tile_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
        %b_tile_ptr = offset %b_ptr, %b_tile_offsets :
            tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
```

### Looping Over Tiles

Now after all that preparation we can perform the core computation of the kernel:

```cuda_tile
        %C_tile, %a_ptr_final, %b_ptr_final = for %k in (%range_start to %k_tile_size, step %range_start) : tile<i32>
            iter_values(
                %acc_prev = %init_accum,
                %a_tile_ptr_prev = %a_tile_ptr,
                %b_tile_ptr_prev = %b_tile_ptr
            ) -> (tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>)
        {
            // Load a single 64x64 matrix from the tile.
            %A_tile, %token_a = load_ptr_tko weak %a_tile_ptr :
                tile<64x64xptr<f32>> -> tile<64x64xf32>, token

            // Load a single 64x64 matrix from the tile.
            %B_tile, %token_b = load_ptr_tko weak %b_tile_ptr :
                tile<64x64xptr<f32>> -> tile<64x64xf32>, token

            %C_tile_acc = mmaf %A_tile, %B_tile, %acc_prev: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

            // Advance by K block size.
            %block_size = constant <i32: 64> : tile<64x64xi32>
            %a_tile_ptr_next = offset %a_tile_ptr_prev, %block_size
                : tile<64x64xptr<f32>>, tile<64x64xi32>
                    -> tile<64x64xptr<f32>>
            %b_tile_ptr_next = offset %b_tile_ptr_prev, %block_size
                : tile<64x64xptr<f32>>, tile<64x64xi32>
                    -> tile<64x64xptr<f32>>

            // Store the partial sum to the 64x64 accumulator.
            continue %C_tile_acc, %a_tile_ptr_next, %b_tile_ptr_next : tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>
        }
```

After completing the reduction over the K dimension we need to store the output tile to the C matrix:

```cuda_tile
        // Compute C tile offsets and store.
        %c_tile_x_start = muli %block_x_index, %k_tile_size : tile<i32>
        %c_tile_x_start_reshape = reshape %c_tile_x_start : tile<i32> -> tile<1xi32>
        %c_tile_x_start_tensor = broadcast %c_tile_x_start_reshape :
            tile<1xi32> -> tile<64xi32>
        %c_tile_x_offsets_vec = addi %c_tile_x_start_tensor, %tile_size_range : tile<64xi32>

        %c_tile_y_start = muli %block_x_index, %k_tile_size : tile<i32>
        %c_tile_y_start_reshape = reshape %c_tile_y_start : tile<i32> -> tile<1xi32>
        %c_tile_y_start_tensor = broadcast %c_tile_y_start_reshape :
            tile<1xi32> -> tile<64xi32>
        %c_tile_y_offsets_vec = addi %c_tile_y_start_tensor, %tile_size_range : tile<64xi32>

        %c_tile_x_offsets_matrix = reshape %c_tile_x_offsets_vec : tile<64xi32> -> tile<64x1xi32>
        %c_tile_x_offsets_broadcast = broadcast %c_tile_x_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
        %c_tile_x_offsets = muli %c_tile_x_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        %c_tile_y_offsets_matrix = reshape %c_tile_y_offsets_vec : tile<64xi32> -> tile<1x64xi32>
        %c_tile_y_offsets_broadcast = broadcast %c_tile_y_offsets_matrix : tile<1x64xi32> -> tile<64x64xi32>
        %c_tile_y_offsets = muli %c_tile_y_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        %c_tile_offsets = muli %c_tile_x_offsets, %c_tile_y_offsets : tile<64x64xi32>

        %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
            tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %c_ptr = broadcast %c_ptr_base_tensor :
            tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
        %c_tile_ptr = offset %c_ptr, %c_tile_offsets :
            tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

        store_ptr_tko weak %c_tile_ptr, %C_tile :
            tile<64x64xptr<f32>>, tile<64x64xf32> -> token
    }
}
```

## Structured Pointers

The previous examples have looked at how to define matrix multiplication using tensors of pointers and scatter/gather style loads and stores, which take arbitrary tensors of pointers to operate on. These operations give maximal expressivity to programmers but at the cost of potential performance.

For example, if a user created a complete disjoint tensor of pointers, it is challenging for a human or a compiler to obtain meaningful performance from this program. In the worst case, each element will become a completely disjoint memory operation, preventing vectorized or tensorized operations, or code which makes use of cache or thread locality.

Tile IR will do its best to obtain good performance with these stores, but optimal performance can more easily be achieved by using a structured pointer, called a **tensor view** in Tile IR. Tensor views allow us to simplify the programming model of Tile IR and to improve the efficiency of user programs.

When constructing a tensor view, we convert the raw pointer into a tensor using static or dynamic shape and stride information, which is done via `cuda_tile.make_tensor_view`. This operation attaches shape and stride information to the pointer and effectively converts a typed pointer to a tensor.

Structured pointers, or tensor views, encapsulate shape and stride information, enabling the compiler to optimize memory access and simplifying the user's code.

## Tiling and Views

Once you have constructed a tensor view, we can make use of `cuda_tile.make_partition_view` to perform tiling of the underlying tensor.

### Vector Addition with Views (SAXPY Example)

We can implement a more complex vector operation with SAXPY or "Single-Precision A . X Plus Y" (a common BLAS operation). We can use tensor view and `cuda_tile.make_partition_view` to implement this operation for arbitrary sized vectors.

The kernel first defines its arguments:

```cuda_tile
entry @saxpy_memref(%X: tile<ptr<f32>>,
                    %Y: tile<ptr<f32>>,
                    %alpha: tile<f32>,
                    %M : tile<i32>,
                    %N : tile<i32>) {
```

Construct tensor views:

```cuda_tile
%x_memref = make_tensor_view %X, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
%y_memref = make_tensor_view %Y, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
```

Create partition views:

```cuda_tile
%x_view = make_partition_view %x_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
%y_view = make_partition_view %y_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
```

Load, compute, and store:

```cuda_tile
%x_tile, %token_x = load_view_tko weak %x_view[%tileIdX, %tileIdY] :
    partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token
%y_tile, %token_y = load_view_tko weak %y_view[%tileIdX, %tileIdY] :
    partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token

// Step 6. Compute sAXPY: y = alpha * A + y
%9 = mulf %alpha_tensor, %x_tile rounding<nearest_even> : tile<128x256xf32>
%result_tile = addf %9, %y_tile rounding<nearest_even> : tile<128x256xf32>

// Step 7. Store the result tile to Y
store_view_tko weak %result_tile, %y_view[%tileIdX, %tileIdY] :
    tile<128x256xf32>, partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> token
```

### Dynamic GEMM with Views

We can put together many of these ideas to support a dynamic GEMM kernel using structured pointers. The inputs are transposed into column-major layout, and the inputs are in fp16 while the output is in fp32.

```cuda_tile
cuda_tile.module @gemm_kloop_module {
    entry @gemm_kloop_kernel(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>, %N: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>, %stride_bn: !cuda_tile.tile<i32>, %stride_cm: !cuda_tile.tile<i32>
    ) {
```

Use `cuda_tile.assume` to inform the compiler about alignment:

```cuda_tile
        %A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
        %B_ptr_assume = assume #cuda_tile.div_by<16>, %B_ptr : tile<ptr<f16>>
        %C_ptr_assume = assume #cuda_tile.div_by<16>, %C_ptr : tile<ptr<f32>>
        %stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>
        %stride_bn_assume = assume #cuda_tile.div_by<8>, %stride_bn : tile<i32>
        %stride_cm_assume = assume #cuda_tile.div_by<8>, %stride_cm : tile<i32>
```

Create tensor views for A, B, and C:

```cuda_tile
        // A reference to the A tensor pointed to by A_ptr, (K x M)
        %A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the B tensor pointed to by B_ptr, (N x K)
        %B = make_tensor_view %B_ptr_assume, shape = [%N, %K], strides = [%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the C tensor pointed to by C_ptr, (M x N)
        %C = make_tensor_view %C_ptr_assume, shape = [%M, %N], strides = [%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
```

Create partition views:

```cuda_tile
        %A_block  = make_partition_view %A : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        %B_block  = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        %C_block  = make_partition_view %C : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>
```

Read tile block grid coordinates:

```cuda_tile
        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
```

Get the dynamic reduction dimension:

```cuda_tile
        %mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>
```

Loop over the K dimension:

```cuda_tile
        %result = for %k in (%i0 to %mk_len_i32#1, step %i1) : tile<i32>
            iter_values(%acc_prev = %cst) -> (tile<128x128xf32>)
        {
            // Load a single 128x64 matrix from the tile.
            %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x64xf16>, token

            // Load a single 64x128 matrix from the tile.
            %B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token

            // Compute the mma(A_frag, B_frag) + acc_prev.
            %acc = mmaf %A_frag, %B_frag, %acc_prev: tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
            // Store the partial sum to the 128x128 accumulator.
            continue %acc : tile<128x128xf32>
        }

        // Finally store the complete 128x128 tile to the view of C.
        %t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
    }
}
```

## Cross TileBlock Communication

The following example demonstrates how tile blocks can communicate through global memory using atomic operations:

```cuda_tile
  cuda_tile.module @hello_cross_block {
    global @_global_printf_mutex <i32: 1> : tile<1xi32>

    entry @hello_cross_block_kernel() {
      %idx, %idy, %idz = get_tile_block_id : tile<i32>
      %tilex, %tiley, %tilez = get_num_tile_blocks : tile<i32>
      %2 = get_global @_global_printf_mutex : tile<ptr<i32>>
      %3 = cuda_tile.constant <i32: 0> : tile<i32>
      %4 = cuda_tile.constant <i32: 1> : tile<i32>
      loop {
        %t1 = make_token : token
        %6, %t2 = atomic_cas_tko relaxed device %2, %4, %3 token=%t1: tile<!cuda_tile.ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
        %7 = trunci %6 : tile<i32> -> tile<i1>
        if %7 {
          break
        }
      }
      print "current tile: %i / %i\0A", %idx, %tilex : tile<i32>, tile<i32>
      %5, %t4 = atomic_rmw_tko relaxed device %2, xchg, %4: tile<ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
      return
    }
  }
```

This example uses a global variable as a mutex with atomic compare-and-swap to ensure only one tile block prints at a time. After printing, the mutex is released using atomic exchange.
