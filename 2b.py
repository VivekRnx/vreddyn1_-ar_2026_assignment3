import numpy as np
import matplotlib.pyplot as plt

# Intrinsics from Q1.b
fx, fy, ox, oy = 600.0, 600.0, 400.0, 300.0
# Lens parameters d = [k1, k2, p1, p2, k3]
k1, k2, p1, p2, k3 = -0.30, 0.12, 0.005, -0.005, 0.01

# Camera-frame points
PC = np.array([[-0.4, -0.4, 2.0],
               [ 0.4, -0.4, 2.0],
               [ 0.4,  0.4, 2.0],
               [-0.4,  0.4, 2.0]])

def project(P, distort):
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    if distort:
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        yd = y*radial + p1*(r2 + 2*y**2) + 2*p2*x*y
    else:
        xd, yd = x, y
    u = fx*xd + ox
    v = fy*yd + oy
    return np.column_stack([u, v])

I0 = project(PC, distort=False)   # without distortion
I1 = project(PC, distort=True)    # with distortion
err = np.linalg.norm(I1 - I0, axis=1)

for i in range(4):
    print(f"P{i+1}: undistorted ({I0[i,0]:.3f}, {I0[i,1]:.3f})  "
          f"distorted ({I1[i,0]:.3f}, {I1[i,1]:.3f})  error {err[i]:.4f} px")
print(f"Sum of errors = {err.sum():.4f} px, average error = {err.mean():.4f} px")

#Plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

# Left: points on the full 800 x 600 image
a = ax[0]
a.plot([0, 800, 800, 0, 0], [0, 0, 600, 600, 0], color="0.6", lw=1)
a.plot(*np.vstack([I0, I0[:1]]).T, "o--", color="#2a78d6", ms=7, lw=1,
       label="without distortion")
a.plot(*np.vstack([I1, I1[:1]]).T, "x-", color="#eb6834", ms=8, lw=1,
       label="with distortion")
a.plot(ox, oy, "+", color="0.3", ms=10, label="principal point (400, 300)")
for i in range(4):
    a.annotate(f"P{i+1}", I0[i], textcoords="offset points", xytext=(-22, -4))
a.set_xlim(-20, 820); a.set_ylim(620, -20)   # image v axis points down
a.set_aspect("equal")
a.set_xlabel("u (pixels)"); a.set_ylabel("v (pixels)")
a.set_title("Q2.b image points")
a.legend(loc="lower right", fontsize=8)

# Right: displacement vectors (distorted - undistorted), magnified 10x
b = ax[1]
s = 10
b.plot(*np.vstack([I0, I0[:1]]).T, "o--", color="#2a78d6", ms=7, lw=1)
for i in range(4):
    d = s*(I1[i] - I0[i])
    b.annotate("", xy=I0[i] + d, xytext=I0[i],
               arrowprops=dict(arrowstyle="->", color="#eb6834", lw=1.5))
    b.annotate(f"P{i+1}: {err[i]:.2f} px", I0[i], textcoords="offset points",
               xytext=(-30, 9 if I0[i, 1] < oy else -16), fontsize=8)
b.plot(ox, oy, "+", color="0.3", ms=10)
b.set_xlim(230, 570); b.set_ylim(460, 140)
b.set_aspect("equal")
b.set_xlabel("u (pixels)"); b.set_ylabel("v (pixels)")
b.set_title("Shift caused by distortion (arrows x10)")

plt.tight_layout()
plt.savefig("q2b_plot.png", dpi=200)
