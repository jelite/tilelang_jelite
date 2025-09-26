import torch

# 랜덤 텐서 생성
B, C, H, L, S, P = 2, 3, 4, 5, 6, 7
scores_decay = torch.randn(B, C, H, L, S)
dt = torch.randn(B, H, C, S)
x = torch.randn(B, C, S, H, P)

# -------------------------
# 1) 파이썬 루프 구현 (ground truth)
# -------------------------
out_loop = torch.zeros(B, C, L, H, P)
for b in range(B):
    for c in range(C):
        for l in range(L):
            for h in range(H):
                for p in range(P):
                    tmp = 0.0
                    for s in range(S):
                        tmp += scores_decay[b,c,h,l,s] * dt[b,h,c,s] * x[b,c,s,h,p]
                    out_loop[b,c,l,h,p] = tmp

# -------------------------
# 2) 두 단계 연산 (loop 버전)
# -------------------------
out_two_step = torch.zeros(B, C, L, H, P)
for b in range(B):
    for c in range(C):
        for h in range(H):
            # (l,s) = scores_decay * dt
            weights = torch.zeros(L, S)
            for l in range(L):
                for s in range(S):
                    weights[l,s] = scores_decay[b,c,h,l,s] * dt[b,h,c,s]

            # (l,s) @ (s,p) = (l,p)
            for l in range(L):
                for p in range(P):
                    tmp = 0.0
                    for s in range(S):
                        tmp += weights[l,s] * x[b,c,s,h,p]
                    out_two_step[b,c,l,h,p] = tmp

# -------------------------
# 3) einsum 직접 실행
# -------------------------
out_einsum = torch.einsum("bchls,bhcs,bcshp->bclhp", scores_decay, dt, x)

# -------------------------
# 4) 결과 비교
# -------------------------
print("loop vs einsum:  ", torch.allclose(out_loop, out_einsum, atol=1e-6))
print("two-step vs einsum:", torch.allclose(out_two_step, out_einsum, atol=1e-6))

print("Output shape:", out_loop.shape)
