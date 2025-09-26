# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def gemm(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
        B = T.match_buffer(B_handle, (1024, 1024), "float16", strides=(1024, 1))
        C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.block("tilelang_root"):
            T.reads(A[by * 128, 0:993], B[0:993, bx * 128], C[by * 128, bx * 128])
            T.writes()
            A_shared = T.alloc_buffer((1, 16, 256), "float16", scope="shared.dyn")
            B_shared = T.alloc_buffer((2, 4, 512), "float16", scope="shared.dyn")
            C_local = T.alloc_buffer((128,), scope="local")
            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                for vec in T.vectorized(2):
                    C_local[i * 2 + vec] = T.float32(0.0)
            for k in T.serial(32, annotations={"num_stages": 3}):
                if tx == 0:
                    T.tma_load(T.create_tma_descriptor(6, 2, A.data, 1024, 1024, T.int64(2), T.int64(2048), 32, 128, 1, 1, 0, 2, 2, 0), 0, T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 4096, 2), k * 32, by * 128, 0)
                if tx == 0:
                    for i in T.unroll(2):
                        T.tma_load(T.create_tma_descriptor(6, 2, B.data, 1024, 1024, T.int64(2), T.int64(2048), 64, 32, 1, 1, 0, 3, 2, 0), 0, T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, i * 2048, 2048, 2), bx * 128 + i * 64, k * 32, 0)
                T.tl_gemm("tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>", T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 16384, 3))
            for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                for vec in T.vectorized(2):
                    C[by * 128 + i // 32 * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4, bx * 128 + i % 32 // 2 * 8 + tx % 4 * 2 + vec] = T.Cast("float16", C_local[i * 2 + vec])