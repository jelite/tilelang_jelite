import torch

# A: (2,3,4), B: (4,3,2)
A = torch.randn(2, 3, 4)
B = torch.randn(4, 3, 2)

# 1) 완전 합산 (스칼라)
out_einsum_scalar = torch.einsum("ijk,kji->", A, B)

out_loop_scalar = 0.0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            out_loop_scalar += A[i,j,k] * B[k,j,i]

print("Scalar:", out_einsum_scalar.item(), out_loop_scalar)

# 2) i만 남김 → (2,)
out_einsum_i = torch.einsum("ijk,kji->i", A, B)

out_loop_i = torch.zeros(A.shape[0])
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            out_loop_i[i] += A[i,j,k] * B[k,j,i]

print("i left match:", torch.allclose(out_einsum_i, out_loop_i))

# einsum 방식
out_einsum = torch.einsum("ijk,kji->ijk", A, B)

# loop 방식
out_loop = torch.zeros_like(A)
for i in range(A.shape[0]):      # 2
    for j in range(A.shape[1]):  # 3
        for k in range(A.shape[2]):  # 4
            out_loop[i,j,k] = A[i,j,k] * B[k,j,i]

print("same?", torch.allclose(out_einsum, out_loop))