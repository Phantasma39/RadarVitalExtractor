import numpy as np
import matplotlib.pyplot as plt


def fit_circle_ransac_iq(z,
                         n_iter=20000,
                         min_inlier_ratio=0.01,  # 这个是有几个点在规定范围内，越大越严格
                         random_state=None):
    scale = np.max(np.abs(z))  # 确定eps值，因为数据很大
    eps = 0.003 * scale  # 这个是误差范围，越小越严格

    rng = np.random.default_rng(random_state)
    pts = np.column_stack([np.real(z), np.imag(z)])
    N = len(pts)

    def circle_from_3pts(p1, p2, p3):  # 找三个点
        A = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1],
        ])
        B = np.array([
            -(p1[0] ** 2 + p1[1] ** 2),
            -(p2[0] ** 2 + p2[1] ** 2),
            -(p3[0] ** 2 + p3[1] ** 2),
        ])
        C = np.linalg.solve(A, B)
        xc = -0.5 * C[0]
        yc = -0.5 * C[1]
        R = np.sqrt(xc ** 2 + yc ** 2 - C[2])
        return xc, yc, R

    # ---- Helper: algebraic least squares circle fit (Kåsa) ----
    def fit_circle_least_squares(P):
        x = P[:, 0]
        y = P[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x ** 2 + y ** 2)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc = -0.5 * c[0]
        yc = -0.5 * c[1]
        R = np.sqrt(xc ** 2 + yc ** 2 - c[2])
        return xc, yc, R

    best_inliers = []
    best_model = None

    for _ in range(n_iter):
        # sample 3 distinct points
        idx = rng.choice(N, 3, replace=False)
        try:
            xc, yc, R = circle_from_3pts(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        except np.linalg.LinAlgError:
            continue

        # compute inliers
        d = np.abs(np.sqrt((pts[:, 0] - xc) ** 2 + (pts[:, 1] - yc) ** 2) - R)
        inliers = pts[d < eps]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (xc, yc, R)

    # ====================== 这里是修改的核心 ======================
    # check minimal support → 不报错，只打印警告，返回 None
    if len(best_inliers) < min_inlier_ratio * N:
        print(f"⚠️  拟合失败：内点比例 {len(best_inliers) / N:.2f} < 最小要求 {min_inlier_ratio}，跳过该通道")
        return None, None, None  # 返回空值，程序不崩溃

    # refine using all inliers
    xc, yc, R = fit_circle_least_squares(best_inliers)

    # ---- Plot IQ data and fitted circle (single plot) ----
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.scatter(pts[:, 0], pts[:, 1], s=1)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), color='red')
    ax.scatter([xc], [yc])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('IQ constellation with fitted circle center')
    ax.grid(True)
    ax.text(xc, yc, f"({xc:.3f}, {yc:.3f})",
            ha='left', va='bottom')

    plt.show()

    return xc, yc, R
