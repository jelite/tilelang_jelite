import tilelang
import tilelang.language as T
from tilelang import tvm
from pathlib import Path
from typing import Optional


def _normalize_stage_name(name: str) -> str:
    sanitized = name.strip().lower().replace(" ", "_")
    for ch in "()-.":
        sanitized = sanitized.replace(ch, "_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _print_ir_stage(name, ir_module, dump_dir: Optional[Path] = None):
    """Utility to keep IR stage dumps easy to spot in stdout and persist them."""
    print(f"\n=== {name} ===")
    print(ir_module.script())

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        stage_file = dump_dir / f"{_normalize_stage_name(name)}.tir.py"
        stage_file.write_text(ir_module.script())


def dump_lowering_stages(kernel):
    """Re-run the lowering pipeline and print the intermediate IR modules."""
    from importlib import import_module

    engine_lower = import_module("tilelang.engine.lower")
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
    import tilelang.transform as tl_transform

    prim_func = kernel.prim_func
    target = kernel.target
    pass_configs = kernel.pass_configs or {}

    kernel_name = prim_func.attrs["global_symbol"]
    dump_dir = Path(__file__).resolve().parent / "ir_dumps" / kernel_name

    mod = tvm.IRModule({kernel_name: prim_func})
    _print_ir_stage("Original TileLang IR", mod, dump_dir)

    optimized_mod = None
    with tl_transform.PassContext(opt_level=3, config=pass_configs):
        with tvm.target.Target(target):
            lowered_mod = LowerAndLegalize(mod, target)
            _print_ir_stage("After LowerAndLegalize", lowered_mod, dump_dir)

            optimized_mod = OptimizeForTarget(lowered_mod, target)
            _print_ir_stage("After OptimizeForTarget", optimized_mod, dump_dir)

    if optimized_mod is None:
        raise RuntimeError("Failed to run OptimizeForTarget during lowering inspection.")

    is_cpu_backend = engine_lower.is_cpu_device_backend(target)
    host_filter = engine_lower.get_host_call(is_device_c=is_cpu_backend)
    device_filter = engine_lower.get_device_call(is_device_c=is_cpu_backend)

    host_mod = tvm.tir.transform.Filter(host_filter)(optimized_mod)
    device_mod = tvm.tir.transform.Filter(device_filter)(optimized_mod)

    _print_ir_stage("Host Module (Post-Split)", host_mod, dump_dir)
    _print_ir_stage("Device Module (Post-Split)", device_mod, dump_dir)


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    dump_lowering_stages(kernel)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
