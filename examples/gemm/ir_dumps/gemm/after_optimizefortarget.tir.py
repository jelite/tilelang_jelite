# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def gemm(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        B_desc = T.handle("uint8x128", "grid_constant")
        B = T.handle("float16", "global")
        A_desc = T.handle("uint8x128", "grid_constant")
        A = T.handle("float16", "global")
        T.func_attr({"dyn_shared_memory_buf": 49152, "target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 8, "blockIdx.y": 8, "threadIdx.x": 256, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_entry_func": True, "tma_descriptor_args": {B_desc: ["__tvm_tensormap_create_tiled", B_desc, 6, 2, B, 1024, 1024, 2, 2048, 64, 32, 1, 1, 0, 3, 2, 0], A_desc: ["__tvm_tensormap_create_tiled", A_desc, 6, 2, A, 1024, 1024, 2, 2048, 32, 128, 1, 1, 0, 2, 2, 0]}})
        A_1 = T.match_buffer(A_handle, (1024, 1024), "float16", data=A, strides=(1024, 1))
        B_1 = T.match_buffer(B_handle, (1024, 1024), "float16", data=B, strides=(1024, 1))
        C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
        with T.LetStmt(T.tvm_stack_alloca("arg_value", 16), var=A_desc):
            T.call_packed("__tvm_tensormap_create_tiled", A_desc, 6, 2, A, 1024, 1024, 2, 2048, 32, 128, 1, 1, 0, 2, 2, 0)
            with T.LetStmt(T.tvm_stack_alloca("arg_value", 16), var=B_desc):
                T.call_packed("__tvm_tensormap_create_tiled", B_desc, 6, 2, B, 1024, 1024, 2, 2048, 64, 32, 1, 1, 0, 3, 2, 0)
                bx = T.launch_thread("blockIdx.x", 8)
                buf_dyn_shmem = T.allocate([49152], "uint8", "shared.dyn")
                C_local = T.allocate([128], "float32", "local")
                by = T.launch_thread("blockIdx.y", 8)
                tx = T.launch_thread("threadIdx.x", 256)
                T.create_barriers(6)
                if T.tl_shuffle_elect(0):
                    T.call_extern("handle", "tl::prefetch_tma_descriptor", A_desc)
                    T.call_extern("handle", "tl::prefetch_tma_descriptor", B_desc)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(0), 128)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(1), 128)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(2), 128)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(3), 128)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(4), 128)
                    T.ptx_init_barrier_thread_count(T.get_mbarrier(5), 128)
                T.tvm_storage_sync("shared")
                ty = T.launch_thread("threadIdx.y", 1)
                tz = T.launch_thread("threadIdx.z", 1)
                T.attr([128, 128], "kWarpSpecializationScope", 0)
                if 128 <= tx:
                    T.set_max_nreg(24, 0)
                    for k in range(32):
                        T.mbarrier_wait_parity(T.get_mbarrier(k % 3 + 3), T.bitwise_xor(k % 6 // 3, 1))
                        if T.tl_shuffle_elect(128):
                            T.mbarrier_expect_tx(T.get_mbarrier(k % 3), 8192)
                            T.tma_load(A_desc, T.get_mbarrier(k % 3), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, k % 3 * 4096, 4096, 2), k * 32, by * 128, 0)
                            T.mbarrier_expect_tx(T.get_mbarrier(k % 3), 8192)
                            T.tma_load(B_desc, T.get_mbarrier(k % 3), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + k % 3 * 4096, 2048, 2), bx * 128, k * 32, 0)
                            T.tma_load(B_desc, T.get_mbarrier(k % 3), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (k % 3 * 4096 + 2048), 2048, 2), bx * 128 + 64, k * 32, 0)
                        T.ptx_arrive_barrier(T.get_mbarrier(k % 3))
                else:
                    T.set_max_nreg(240, 1)
                    C_local_1 = T.Buffer((128,), data=C_local, scope="local")
                    for i in T.unroll(64):
                        C_local_1[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0.0), 2)
                    T.fence_proxy_async()
                    for k in range(32):
                        T.mbarrier_wait_parity(T.get_mbarrier(k % 3), k % 6 // 3)
                        T.tl_gemm("tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, k % 3 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + k % 3 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local, 0, 16384, 3))
                        T.ptx_arrive_barrier(T.get_mbarrier(k % 3 + 3))
                    for i in T.unroll(64):
                        C_1 = T.Buffer((1048576,), "float16", data=C.data)
                        C_1[by * 131072 + i // 32 * 65536 + tx // 32 * 16384 + i % 2 * 8192 + tx % 32 // 4 * 1024 + bx * 128 + i % 32 // 2 * 8 + tx % 4 * 2:by * 131072 + i // 32 * 65536 + tx // 32 * 16384 + i % 2 * 8192 + tx % 32 // 4 * 1024 + bx * 128 + i % 32 // 2 * 8 + tx % 4 * 2 + 2] = T.Cast("float16x2", C_local_1[i * 2:i * 2 + 2])