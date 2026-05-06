# Appendix

## Programming Model Example Programs

### Hello Tile Block

```cuda_tile
cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}
```

### Vector Addition Block 128x1

A basic implementation of 128 sized vector addition using unstructured load/stores.

This implements addition over a 1-d tensor (vector) with size 128.

128x1 + 128x1 => 128x1

```cuda_tile
cuda_tile.module @vector_block_add_128x1 {
    entry @vector_block_add_128x1_kernel(
        %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)
{
    // Create an offset on the inclusive (0, 127) interval.
    %offset = iota : tile<128xi32>
    // We need a tile<ptr<T>> in order to perform a load or store.
    //
    // We will now convert each raw base pointer into such a pointer.
    //
    // First reshape the scalar pointer ptr<f32> to tile<1xptr<f32>> so it has the correct rank.
    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    // Next broadcast the pointer so we have a tensor of (base, ..., base) containing 128 elements.
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    // Finally add the offset tensor to the tensor of pointers to obtain a tile<128xptr<f32>> that contains
    // pointers of (base + 0, ..., base + 127) as its values.
    %a_tensor = offset %a_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now we do the same for B.
    %b_ptr_base_tensor =reshape %b_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %b_ptr = broadcast %b_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %b_tensor = offset %b_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // And the same for C.
    %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %c_ptr = broadcast %c_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %c_tensor = offset %c_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now that we have prepared all the pointers we can do the real work.
    //
    // First we load A, and B into %a_val and %b_val.
    %a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    %b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    // We then compute floating-point vector addition using addf
    %c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
    // Finally we store the result to C.
    store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
  }
}
```

### Hello Tile Grid

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

### GEMM Single 64x64 Block

An implementation of GEMM for a single statically shaped square 64x64 block.

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

### GEMM 4096x4096 Block

An implementation of GEMM for a square 4096x4096 @ 4096x4096 multiplication.

This would be launched on a 64x64 grid where each tile block computes 64x64 output tiles of C.

```cuda_tile
cuda_tile.module @gemm_square_4096_tile_64x64_module {
    entry @gemm_square_4096_tile_64x64_kernel(
        %a_ptr_base_scalar: tile<ptr<f32>>,
        %b_ptr_base_scalar: tile<ptr<f32>>,
        %c_ptr_base_scalar: tile<ptr<f32>>
    ) {
        // We first setup up some state for the kernel.
        // Read Tile block id's.
        %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

        // We assume we have tiled a 4096x4096 @ 4096x4096 matrix multiplication split into
        // 64x64 tiles so the tile m, n, k are all 64.
        %m_tile_size = constant <i32: 64> : tile<i32>
        %m_stride_factor = cuda_tile.constant <i32: 64> : tile<64x64xi32>
        %k_tile_size = cuda_tile.constant <i32: 64> : tile<i32>

        %range_start = cuda_tile.constant <i32: 0> : tile<i32>
        %range_step = cuda_tile.constant <i32: 1> : tile<i32>
        %init_accum = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>

        // The shared range from (0, 63).
        %tile_size_range = cuda_tile.iota : tile<64xi32>

        // We must first compute the tensors of initial offsets for A and B so that we can obtain a tensor of
        // pointers for each to load them.
        %a_tile_base = cuda_tile.muli %block_x_index, %k_tile_size : tile<i32>
        %a_tile_base_reshape = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
        %a_tile_base_tensor = cuda_tile.broadcast %a_tile_base_reshape :
            tile<1xi32> -> tile<64xi32>
        %m_offsets_vec = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<64xi32>

        // We first broadcast the m_offsets into a matrix where each column is identical and scaled by stride.
        %m_offsets_matrix = cuda_tile.reshape %m_offsets_vec :
            tile<64xi32> -> tile<64x1xi32>
        %m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix :
            tile<64x1xi32> -> tile<64x64xi32>
        %m_offsets = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        // We then broadcast the k_offsets into a matrix where row is identical and scaled by stride.
        %ak_offsets_matrix = cuda_tile.reshape %tile_size_range :
             tile<64xi32> -> tile<1x64xi32>
        %ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix :
            tile<1x64xi32> -> tile<64x64xi32>
        %ak_offsets = cuda_tile.muli %ak_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        // Finally we add them together resulting in the final offset matrix for A.
        %a_tile_offsets = cuda_tile.addi %m_offsets, %ak_offsets : tile<64x64xi32>

        // Now we do the same for B, first prepare the set of n_offsets.
        %b_tile_base = cuda_tile.muli %block_y_index, %k_tile_size : tile<i32>
        %b_tile_base_reshape = cuda_tile.reshape %b_tile_base :
            tile<i32> -> tile<1xi32>
        %b_tile_base_tensor = cuda_tile.broadcast %b_tile_base_reshape :
            tile<1xi32> -> tile<64xi32>
        %n_offsets_vec = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<64xi32>

        // b_tile = k_offs[:, None] * k_stride + n_offs[None, :] * n_stride
        %bk_offsets_matrix = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
        %bk_offsets = cuda_tile.broadcast %bk_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>

        %n_offsets_matrix = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
        %n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix :  tile<1x64xi32> -> tile<64x64xi32>
        %n_offsets = cuda_tile.muli %n_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

        %b_tile_offsets = cuda_tile.muli %bk_offsets, %n_offsets : tile<64x64xi32>

        // Convert base pointers to tensors, add the offset matrix.
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

        // Compute offsets for C.
        %c_tile_x_start = muli %block_x_index, %k_tile_size : tile<i32>
        %c_tile_x_start_reshape = reshape %c_tile_x_start : tile<i32> -> tile<1xi32>
        %c_tile_x_start_tensor = broadcast %c_tile_x_start_reshape : tile<1xi32> -> tile<64xi32>
        %c_tile_x_offsets_vec = addi %c_tile_x_start_tensor, %tile_size_range : tile<64xi32>

        %c_tile_y_start = muli %block_x_index, %k_tile_size : tile<i32>
        %c_tile_y_start_reshape = reshape %c_tile_y_start : tile<i32> -> tile<1xi32>
        %c_tile_y_start_tensor = broadcast %c_tile_y_start_reshape : tile<1xi32> -> tile<64xi32>
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

### Vector Addition with tensor_view

Tiled SAXPY is an optimized implementation of the SAXPY operation using tensor_view abstractions.

```cuda_tile
cuda_tile.module @saxpy {
    // TileIR kernel function
    entry @saxpy_memref(%X: tile<ptr<f32>>,
                        %Y: tile<ptr<f32>>,
                        %alpha: tile<f32>,
                        %M : tile<i32>,
                        %N : tile<i32>) {

        // Step 1. Get the tile block ID
        %tileIdX, %tileIdY, %tileIdZ = get_tile_block_id : tile<i32>

        // Step 2. Reshape and broadcast the alpha scalar
        %alpha_reshaped = reshape %alpha : tile<f32> -> tile<1x1xf32>
        %alpha_tensor = broadcast %alpha_reshaped : tile<1x1xf32> -> tile<128x256xf32>

        // Step 3. Create tensor_view for X and Y
        %x_memref = make_tensor_view %X, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
        %y_memref = make_tensor_view %Y, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // Step 4. Create partition view for X and Y
        %x_view = make_partition_view %x_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
        %y_view = make_partition_view %y_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>

        // Step 5. Load tile from X and Y
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
    }
}
```

### GEMM Tiled with tensor_view

An implementation of GEMM using tensor_view with dynamic sizes.

Kernel computes MxNxK with 128x128x64 Tile Size. Computes F32 += f16 * f16 + 0.0.

```cuda_tile
cuda_tile.module @gemm_kloop_module {
    entry @gemm_kloop_kernel(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>, %N: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>, %stride_bn: !cuda_tile.tile<i32>, %stride_cm: !cuda_tile.tile<i32>
    ) {
        // First we need to prepare the inputs for the actual computation.
        // Assume the preconditions of this kernel (i.e., the strides are all divisible by 8)
        %A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
        %B_ptr_assume = assume #cuda_tile.div_by<16>, %B_ptr : tile<ptr<f16>>
        %C_ptr_assume = assume #cuda_tile.div_by<16>, %C_ptr : tile<ptr<f32>>
        %stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>
        %stride_bn_assume = assume #cuda_tile.div_by<8>, %stride_bn : tile<i32>
        %stride_cm_assume = assume #cuda_tile.div_by<8>, %stride_cm : tile<i32>

        // Constants must be allocated explicitly in the program.
        %i0 = constant <i32: 0> : !cuda_tile.tile<i32>
        %i1 = constant <i32: 1> : !cuda_tile.tile<i32>
        %cst = constant <f32: 0.000000e+00> : !cuda_tile.tile<128x128xf32>

        // Convert the unstructured pointers `ptr` to `tensor_view`.
        // A reference to the A tensor pointed to by A_ptr, (K x M)
        %A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the B tensor pointed to by B_ptr, (N x K)
        %B = make_tensor_view %B_ptr_assume, shape = [%N, %K], strides = [%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the C tensor pointed to by C_ptr, (M x N)
        %C = make_tensor_view %C_ptr_assume, shape = [%M, %N], strides = [%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // Now we have all the inputs as structured pointers each associated with layouts.
        // Next we will tile the problem.
        // Our matrix multiplication is (M*K) @ (K*N) = M*N but our input tensors are transposed.
        // In order to handle this we create partition view where we flip the 0th and 1st dims.

        // We are blocking A (K x M) -> block_m x block_k.
        %A_block  = make_partition_view %A : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // We are blocking B (N x K) -> block_k x block_n.
        %B_block  = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // We are blocking C (M x N) -> block_m x block_n.
        %C_block  = make_partition_view %C : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>

        // Read Tile block id's.
        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

        // Because we allow for dynamic dimensions we must get the reduction dimension `K` dynamically.
        %mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>

        // Now that we have done all the setup, we can finally perform the computation itself.
        // We simply loop over the K dimension computing: dot(A_block[0, k], B_block[k, 0]).
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

---

## Operation Examples

### cuda_tile.cat_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
  %arg0 = constant <f32: 0.0> : tile<2x4xf32>
  %arg1 = constant <f32: 1.0> : tile<2x4xf32>

     // A valid invocation of cat.
     %0 = cat %arg0, %arg1 dim = 1
       : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

     // >>> %arg0 = tile([[ A, B, C ],
     //                   [ D, E, F ]])
     // >>> %arg1 = tile([[ 1, 2, 3 ],
     //                   [ 4, 5, 6 ]])
     // >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
     //                [ D, E, F, 4, 5, 6 ]])

     // A valid invocation of cat.
     %1 = cat %arg0, %arg1 dim = 0
       : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

     // >>> %arg0 = tile([[ A, B, C ],
     //                   [ D, E, F ]])
     //
     // >>> %arg1 = tile([[ 1, 2, 3 ],
     //                   [ 4, 5, 6 ]])
     //
     // >>> %1 = tile([[ A, B, C ],
     //                [ D, E, F ],
     //                [ 1, 2, 3 ],
     //                [ 4, 5, 6 ]])
  }
}
```

### cuda_tile.constant_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
   %c0 = constant <i32: 0> : tile<i32>
   %c1 = constant <i64: 1> : tile<i64>
   %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
   %c3 = constant <f32: 0.0> : tile<2x4xf32>
   %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
 }
}
```

### cuda_tile.extract_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     // Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
     %c1 = constant <i32: 1> : tile<i32>
     %c2 = constant <i32: 2> : tile<i32>
     %t = constant <f32: 0.0> : tile<32x8xf32>
     // Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
     %0 = extract %t[%c1, %c2]
         : tile<32x8xf32> -> tile<4x2xf32>
  }
}
```

### cuda_tile.get_global_0

```cuda_tile
cuda_tile.module @module {
   global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

   entry @example() {
     %ptr = get_global @val : tile<ptr<f32>>
     return
   }
}
```

### cuda_tile.get_num_tile_blocks_0

```cuda_tile
cuda_tile.module @module {
 entry @example() {
   %x, %y, %z = get_num_tile_blocks : tile<i32>
 }
}
```

### cuda_tile.global_0

```cuda_tile
cuda_tile.module @module {
   global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
   entry @example() {}
}
```

### cuda_tile.permute_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
     %0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
  }
}
```

### cuda_tile.reduce_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %input = constant <f32: 0.0> : tile<8xf32>
     %0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
       (%input_arg: tile<f32>, %input_accum: tile<f32>) {
         %add_result = addf %input_arg, %input_accum : tile<f32>
         yield %add_result : tile<f32>
       }
  }
}
```

### cuda_tile.reduce_1

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %input = constant <f32: 0.0> : tile<8x64xf32>
     %0 = reduce %input dim=1 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
       (%input_arg: tile<f32>, %input_accum: tile<f32>) {
         %add_result = addf %input_arg, %input_accum : tile<f32>
         yield %add_result : tile<f32>
       }
  }
}
```

### cuda_tile.reshape_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %cst = constant <i8: 0> : tile<i8>
     %0 = reshape %cst
         : tile<i8> -> tile<1x1x1xi8>

     %t = constant <f32: 0.0> : tile<8x2xf32>
     %1 = reshape %t
         : tile<8x2xf32> -> tile<2x2x4x1xf32>
  }
}
```

### cuda_tile.reshape_1

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]>
         : tile<2x4xi32>
     %r0 = reshape %cst
   : tile<2x4xi32> -> tile<2x2x2xi32>

   // Step 1: Turn source into 1D tile. Use row-major by convention.
   // %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
   %tmp = reshape %cst
       : tile<2x4xi32> -> tile<8xi32>

   // Step 2: Turn 1D tile into result tile. Use row-major by convention.
   // %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
   %r1 =  reshape %tmp
           : tile<8xi32> -> tile<2x2x2xi32>

  }
}
```

### cuda_tile.scan_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
    %input = constant <f32: 0.0> : tile<8x16xf32>
    %result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
    (%acc: tile<f32>, %elem: tile<f32>) {
      %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
      yield %prod : tile<f32>
    }
   }
  }
```

### cuda_tile.assert_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%arg0: tile<i1>) {
     assert %arg0, "assertion failed" : tile<i1>
  }
}
```

### cuda_tile.break_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
   // Break from the body of a loop.
   loop {
       break
   }

   // Break from an if nested within the loop.
   loop  {
       %condition = constant <i1: 1> : tile<i1>
       if %condition  {
           break
       }
       // ...
   }

   %initValue0 = constant <f32: 0.0> : tile<f32>
   // Break from an if nested within the loop, while yielding values.
   %results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
       %condition = constant <i1: 1> : tile<i1>
       if %condition  {
           // ...
           yield
       } else {
           // %if.loopValue0 = ...
           %loopValue0 = constant <f32: 1.0> : tile<f32>
           break %loopValue0 : tile<f32>
       }
       %loopValue1 = constant <f32: 1.0> : tile<f32>
       continue %loopValue1 : tile<f32>
   }
  }
}
```

### cuda_tile.continue_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lowerBound = constant <i32: 0> : tile<i32>
     %upperBound = constant <i32: 10> : tile<i32>
     %step = constant <i32: 1> : tile<i32>
     %condition = constant <i1: 1> : tile<i1>
     // Continue from the body of a loop.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         continue
     }

     // Continue from an if nested within the loop.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         if %condition  {
             continue
         }
         // ...
     }

   // Continue from an if nested within the loop, while yielding values.
   %initVar0 = constant <f32: 0.0> : tile<f32>
   %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
             iter_values(%var0 = %initVar0) -> (tile<f32>)
     {
         if %condition {
             // ...
             yield
         } else {
             %loopValue0 = constant <f32: 1.0> : tile<f32>
             continue %loopValue0 : tile<f32>
         }
         %loopValue1 = constant <f32: 1.0> : tile<f32>
         continue %loopValue1 : tile<f32>
     }
  }
}
```

### cuda_tile.for_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lowerBound = constant <i32: 0> : tile<i32>
     %upperBound = constant <i32: 10> : tile<i32>
     %step = constant <i32: 1> : tile<i32>

     // A simple loop iterating over an i32 range.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         continue
     }

     %initVal0 = constant <f32: 0.0> : tile<f32>
     // A similar loop to the above, but with a loop carried value, val0.
     %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                         iter_values(%val00 = %initVal0) -> (tile<f32>) {
       %loopVal0 = constant <f32: 1.0> : tile<f32>
       continue %loopVal0 : tile<f32>
     }
  }
}
```

### cuda_tile.if_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %condition = constant <i1: 1> : tile<i1>

     // A simple if operation that conditionally executes a region.
     if %condition  {
       // ...
     }

     // An if operation with an "else" branch.
     if %condition  {
       // ...
     } else {
       // ...
     }

     // An if operation that returns mixed types (f32,i32)
     %x, %y = if %condition -> (tile<f32>, tile<i32>) {
       %x_then = constant <f32: 1.0> : tile<f32>
       %y_then = constant <i32: 2> : tile<i32>
       yield %x_then, %y_then : tile<f32>, tile<i32>
     } else {
       %x_then = constant <f32: 1.0> : tile<f32>
       %y_then = constant <i32: 42> : tile<i32>
       yield %x_then, %y_then : tile<f32>, tile<i32>
     }
  }
}
```

### cuda_tile.loop_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     // A simple "while-do" loop.
     loop {
         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             continue
         }
         break
     }
  }
}
```

### cuda_tile.loop_1

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     // A simple "do-while" loop.
     loop {
         //... body of the loop.

         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             continue
         }
         break
     }
  }
}
```

### cuda_tile.loop_2

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %initValue0 = constant <f32: 0.0> : tile<f32>
     // A loop that yields carried-iteration values, returning the final values.
     %results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             %loopValue0 = constant <f32: 0.0> : tile<f32>
             continue %loopValue0 : tile<f32>
         }
         break %value0 : tile<f32>
     }
  }
}
```

### cuda_tile.loop_3

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %initValue0 = constant <i32: 0> : tile<i32>
     // A loop that uses loop-carried values and returns a different type.
     %results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
         %cond = constant <i1: 1> : tile<i1>

         if %cond {
             %newLoopValue = constant <i32: 0> : tile<i32>
             continue %newLoopValue : tile<i32>
         }

         %finalReturnValue = constant <f32: 0.0> : tile<f32>
         break %finalReturnValue : tile<f32>
     }
  }
}
```

### cuda_tile.return_0

```cuda_tile
cuda_tile.module @module {
   entry @foo() {
     %0 = constant <i32: 0> : tile<i32>
     %1 = constant <f16: 0.0> : tile<f16>
     // ...
     return
   }
}
```

### cuda_tile.yield_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %condition = constant <i1: true> : tile<i1>
     // Yield from the body of an if conditional.
     if %condition  {
         yield
     }

     // Yield values from within an if conditional.
     %x, %y = if %condition -> (tile<f32>, tile<f32>) {
         %x_then = constant <f32: 0.0> : tile<f32>
         %y_then = constant <f32: 1.0> : tile<f32>
         yield %x_then, %y_then : tile<f32>, tile<f32>
     } else {
         %x_else = constant <f32: 2.0> : tile<f32>
         %y_else = constant <f32: 3.0> : tile<f32>
         yield %x_else, %y_else : tile<f32>, tile<f32>
     }
  }
}
```

### cuda_tile.load_ptr_tko_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>) {
     %mask = constant <i1: 1> : tile<i1>
     %padding = constant <f32: 0.0> : tile<f32>

       // Load without token.
       %result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
           : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

       // Load with token.
       %token0 = make_token : token
       %result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
           : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

       return
  }
}
```

### cuda_tile.atan2_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_atan2() {
   %x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
   %y = constant <f32: [1.0,  1.0, 1.0, 0.0]> : tile<4xf32>
   %res = atan2 %x, %y : tile<4xf32>
  }
}
```

### cuda_tile.ceil_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
    %source = constant <f32: 0.5> : tile<f32>
   %result = ceil %source : tile<f32>
  }
}
```

### cuda_tile.cmpf_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example() {
     %lhs0 = constant <f16: 0.0> : tile<f16>
     %rhs0 = constant <f16: 0.0> : tile<f16>

     // Custom form of scalar "ordered equal" comparison.
     %x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

     %lhs1 = constant <f16: 0.0> : tile<2x2xf16>
     %rhs1 = constant <f16: 0.0> : tile<2x2xf16>

     // Custom form of scalar "unordered less than" comparison.
     %x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

     %lhs2 = constant <f64: 0.0> : tile<2x2xf64>
     %rhs2 = constant <f64: 0.0> : tile<2x2xf64>
  }
}
```

### cuda_tile.cos_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_cos() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = cos %in : tile<4xf32>
  }
}
```

### cuda_tile.exp2_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_exp2() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = exp2 %in : tile<4xf32>
  }
}
```

### cuda_tile.exp_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_exp() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = exp %in : tile<4xf32>
  }
}
```

### cuda_tile.floor_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 1.5> : tile<f32>
     %result = floor %source : tile<f32>
  }
}
```

### cuda_tile.log2_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_log2() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = log2 %in : tile<4xf32>
  }
}
```

### cuda_tile.maxf_0

```cuda_tile
cuda_tile.module @module {
    entry @example_maxf(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from partition views.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       // IEEE 754-2019's maximum
       %4 = maxf %2, %3 propagate_nan : tile<2x4xf32>
       // IEEE 754-2019's maximumNumber
       %5 = maxf %2, %3 : tile<2x4xf32>
       // flush denormal to positive zero
       %6 = maxf %2, %3 flush_to_zero : tile<2x4xf32>
  }
}
```

### cuda_tile.minf_0

```cuda_tile
cuda_tile.module @module {
    entry @example_minf(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from partition views.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       // IEEE 754-2019's minimum
       %4 = minf %2, %3 propagate_nan : tile<2x4xf32>
       // IEEE 754-2019's minimumNumber
       %5 = minf %2, %3 : tile<2x4xf32>
       // flush denormal to positive zero
       %6 = minf %2, %3 flush_to_zero : tile<2x4xf32>
  }
}
```

### cuda_tile.mmaf_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lhs0 = constant <f16: 0.0> : tile<4x8xf16>
     %rhs0 = constant <f16: 0.0> : tile<8x2xf16>
     %acc0 = constant <f32: 0.0> : tile<4x2xf32>

     %0 = mmaf %lhs0, %rhs0, %acc0
         : tile<4x8xf16>, tile<8x2xf16>,
           tile<4x2xf32>

     %lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
     %rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
     %acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

     %1 = mmaf %lhs1, %rhs1, %acc1
         : tile<2x4x8xf16>, tile<2x8x2xf16>,
           tile<2x4x2xf32>
  }
}
```

### cuda_tile.negf_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 0.0> : tile<4xf32>
     %result = negf %source : tile<4xf32>
  }
}
```

### cuda_tile.pow_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 0.0> : tile<4xf32>
     %exponent = constant <f32: 2.0> : tile<4xf32>
     %result = pow %source, %exponent : tile<4xf32>
  }
}
```

### cuda_tile.rsqrt_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_rsqrt() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = rsqrt %in : tile<4xf32>

   // Rsqrt op with flush to zero modifier
   %ftz_res = rsqrt %in flush_to_zero : tile<4xf32>
  }
}
```

### cuda_tile.sin_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_sin() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = sin %in : tile<4xf32>
  }
}
```

### cuda_tile.tanh_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_tanh() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res0 = tanh %in : tile<4xf32>

   // tanh with approx modifier
   %res1 = tanh %in rounding<approx> : tile<4xf32>
  }
}
```

### cuda_tile.cmpi_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lhs0 = constant <i16: 0> : tile<i16>
     %rhs0 = constant <i16: 0> : tile<i16>

     // Scalar "signed less than" comparison.
     %x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

     %lhs1 = constant <i64: 0> : tile<2x2xi64>
     %rhs1 = constant <i64: 0> : tile<2x2xi64>

     // Tile equality comparison.
     // There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
     %x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
  }
}
```

### cuda_tile.maxi_0

```cuda_tile
cuda_tile.module @module {
    entry @example_maxi(%arg0: tile<ptr<i32>>, %arg1: tile<ptr<i32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from them.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       // Signless i32 treated as unsigned
       %4 = maxi %2, %3 unsigned : tile<2x4xi32>
       // Signless i32 treated as signed
       %5 = maxi %2, %3 signed : tile<2x4xi32>
  }
}
```

### cuda_tile.mini_0

```cuda_tile
cuda_tile.module @module {
    entry @example_mini(%arg0: tile<ptr<i32>>, %arg1: tile<ptr<i32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from partition views.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       // Signless i32 treated as unsigned
       %4 = mini %2, %3 unsigned : tile<2x4xi32>
       // Signless i32 treated as signed
       %5 = mini %2, %3 signed : tile<2x4xi32>
  }
}
```

### cuda_tile.mmai_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
     %rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
     %acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

     %0 = mmai %lhs0, %rhs0, %acc0 signed signed
         : tile<4x8xi8>, tile<8x2xi8>,
           tile<4x2xi32>

     %lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
     %rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
     %acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

     %1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
         : tile<2x4x8xi8>, tile<2x8x2xi8>,
           tile<2x4x2xi32>
  }
}
```

### cuda_tile.mulhii_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     // 2^31 * 2 = 2^32, or 0x100000000.
     // The most significant 32 bits of the product are 0x00000001.
     // The lower 32 bits of the product are 0x00000000.
     %a = constant <i32: 2147483648> : tile<i32>  // %a = 2^31
     %b = constant <i32: 2> : tile<i32>           // %b = 2
     %res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
     %res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
  }
}
```

### cuda_tile.negi_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
     %result = negi %source : tile<4xi16>
     // %result = [0, -1, -2, -3]
  }
}
```

### cuda_tile.xori_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
     %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
     %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
     // This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
     // are tiles of shape `4xi32`, and returns the result as `%result`.
     %result = xori %lhs, %rhs : tile<4xi32>
  }
}
```

### cuda_tile.atomic_cas_tko_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example(%ptr: tile<ptr<i32>>) {
   %ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
   %ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
   %offsets = iota : tile<8xi32>
   %ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
   %cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
   %val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>
   %mask = constant <i1: [0, 1, 0, 1, 0, 1, 0, 1]> : tile<8xi1>

   // Atomic CAS without input token.
   %0, %token = atomic_cas_tko relaxed device %ptrs, %cmp, %val :
     tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

   // Atomic CAS without input token but with mask.
   %1, %token1 = atomic_cas_tko relaxed device %ptrs, %cmp, %val, %mask :
     tile<8xptr<i32>>, tile<8xi32>, tile<8xi1> -> tile<8xi32>, token

   // Atomic CAS with input token.
   %token2 = make_token : token
   %2, %token3 = atomic_cas_tko relaxed device %ptrs, %cmp, %val token=%token2 :
     tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token

   return
  }
}
```

### cuda_tile.atomic_rmw_tko_0

```cuda_tile
cuda_tile.module @ex_module {
  entry @example_rmw(%ptr: tile<ptr<f32>>) {
   // Reshape the input pointer tile to have a 1d shape
   %ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
   // Broadcast the reshaped tile to a tile with 8 rows, effectively replicating the pointer 8 times
   %ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
   // Create a tile of offsets [0, 1, 2, ..., 7] to index into memory
   %offsets = iota : tile<8xi32>
   // Add the offsets to each pointer in the vector to create 8 unique pointers
   %ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
   %vals = constant <f32: [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]> : tile<8xf32>

   // Perform atomic addf operations on the memory locations pointed by %ptrs
   // without requiring an input token. Returns the original values and a result token
   %0, %res_token0 = atomic_rmw_tko relaxed device %ptrs, addf, %vals :
       tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token

   // Perform atomic add operations again, this time using the explicit input token
   %token = make_token : token
   %1, %res_token1 = atomic_rmw_tko relaxed device %ptrs, addf, %vals, token = %token :
       tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
  }
}
```

### cuda_tile.get_index_space_shape_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
   %tensor_view = make_tensor_view %base,
       shape = [2, 2, 4], strides = [2, 2, 1]
       : tensor_view<2x2x4xf32, strides=[2,2,1]>
   %partition_view = make_partition_view %tensor_view :
     partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>>
   %dim0, %dim1, %dim2 = get_index_space_shape %partition_view :
     partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>> -> tile<i64>
  }
}
```

### cuda_tile.get_tensor_shape_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
    %tensor_view = make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tensor_view<32x32xf32, strides=[32,1]>
   %dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
  }
}
```

### cuda_tile.load_view_tko_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>, %index: tile<i32>) {
     %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
       : tensor_view<8192x128xf32, strides=[128,1]>

     // This example uses the PartitionView on a 8192x128xf32 tensor_view,
     // dividing the tensor_view in tiles of 64x64.

     %view = make_partition_view %tensor_view : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

     %c0 = constant <i32: 0> : tile<i32>
     %c1 = constant <i32: 1> : tile<i32>

     // Load a tile at index (0, 0) in the view's index space.
     // For this PartitionView, this is the rectangular tile such that
     // X=[0,64) and Y=[0,64), in the coordinates of tiles.
     %tile0, %res_token0 = load_view_tko weak %view[%c0, %c0]
       : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

     // Load a tile at index (0, 1) in the view's index space.
     // For this PartitionView, this is the rectangular tile such that
     // X=[0,64) and Y=[64,128), in the coordinates of tiles.
     %tile1, %res_token1 = load_view_tko weak %view[%c0, %c1]
       : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

     // Same example as above but with memory token as input.
     %token = make_token : token
     %tile2, %res_token2 = load_view_tko weak %view[%c0, %c1] token = %token
       : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

     // Loads a tile at the dynamic index (%index, %index) in the view's index space.
     %tile3, %res_token3 = load_view_tko weak %view[%index, %index]
       : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token
  }
}
```

### cuda_tile.make_partition_view_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>) {

     %tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
       : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

     // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
     // the provided tensor_view.
     make_partition_view %tensor_view0 :
       partition_view<
         tile=(1024x1x32),
         tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
       >

     %s0 = constant <i32: 8192> : tile<i32>
     %str0 = constant <i32: 524288> : tile<i32>

     %tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
       : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

     // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
     // the provided tensor_view. The provided tensor_view has a
     // dynamically-sized dimension.
     make_partition_view %tensor_view1 :
       partition_view<tile=(1024x1x32), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
  }
}
```

### cuda_tile.make_tensor_view_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
     // tensor_view to a scalar tile of f32
     %a0 = make_tensor_view %base,
         shape = [], strides = [] : tensor_view<f32>

     // tensor_view to a tile of static shape and strides
     %a1 = make_tensor_view %base,
         shape = [32, 32], strides = [32, 1]
         : tensor_view<32x32xf32, strides=[32,1]>

   %sh0 = constant <i32: 32> : tile<i32>
   %sh1 = constant <i32: 32> : tile<i32>
   %st0 = constant <i32: 32> : tile<i32>
   %st1 = constant <i32: 1> : tile<i32>

     // tensor_view to a tile with partially dynamic shape and strides
     // all dynamic values must be of the same type, here tile<i32>
     %a2 = make_tensor_view %base,
             shape = [%sh0, %sh1], strides = [%st0, %st1]
             : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
  }
}
```

### cuda_tile.store_view_tko_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>) {
     %tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128,1] :
       tensor_view<8192x128xf32, strides=[128,1]>

     // This example uses the PartitionView on a 8192x128xf32 tensor_view,
     // dividing the tensor_view in tiles of 64x64.
     %view = make_partition_view %tensor_view :
       partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

     %c0 = constant <i32: 0> : tile<i32>
     %c1 = constant <i32: 1> : tile<i32>

     %tile = constant <f32: 0.0> : tile<64x64xf32>

     // Store a tile at index (0, 0) in the view's index space.
     // For this TilePartitionView, this is the rectangular tile such that
     // X=[0,64) and Y=[0,64), in the coordinates of tiles.
     %res_token0 = store_view_tko weak %tile, %view[%c0, %c0]
       : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

     // Store a tile at index (0, 1) in the view's index space.
     // For this PartitionView, this is the rectangular tile such that
     // X=[0,64) and Y=[64,128), in the coordinates of tiles.
     %res_token1 = store_view_tko weak %tile, %view[%c0, %c1]
       : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

     // Same example as above but with input token.
     %token = make_token : token
     %res_token2 = store_view_tko weak %tile, %view[%c0, %c1] token = %token
       : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token
    }
  }
```

### cuda_tile.assume_0

```cuda_tile
cuda_tile.module @module {
  entry @example(%input: tile<ptr<f32>>) {
   // Assume that all integers are divisible by 32.
   %int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
   %div_by_1 = assume div_by<32>, %int_tile : tile<8xi16>

   // Assume that every 4th element (starting with element 0) along
   // dimension 0 is divisible by 32 that and all integers are
   // monotonically increasing by 1 within each group of 4.
   %int_tile_2 = constant <i16: [96, 97, 98, 99, 64, 65, 66, 67]> : tile<8xi16>
   %div_by_2 = assume div_by<32, every 4 along 0>, %int_tile_2 : tile<8xi16>

   // Assume that every rectangular chunk of size [1, 4, 2] has the same
   // values.
    %input_rank3 = reshape %input : tile<ptr<f32>> -> tile<1x1x1xptr<f32>>
    %ptr_3d = broadcast %input_rank3 : tile<1x1x1xptr<f32>> -> tile<1x8x8xptr<f32>>
   %same_elem = assume same_elements<[1, 4, 2]>, %ptr_3d : tile<1x8x8xptr<f32>>

   // Assume that every value is greater or equal to 5.
   %int_tile_3 = constant <i16: [5, 9, 10, 11, 6, 5, 5, 7]> : tile<8xi16>
   %bounded = assume bounded<5, ?>, %int_tile_3 : tile<8xi16>
  }
}
```

### cuda_tile.print_tko_0

```cuda_tile
cuda_tile.module @module {
  entry @example() {
      %arg = constant <f32: 0.0> : tile<4xf32>
     print_tko "Hello world: %f\n", %arg : tile<4xf32> -> token
     print_tko "%+08.3f", %arg : tile<4xf32> -> token
  }
}
```
