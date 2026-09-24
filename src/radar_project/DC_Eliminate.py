import os
import time

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免在无显示环境下弹窗/报错
import matplotlib.pyplot as plt
import numpy as np


def fit_circle_ransac_iq(z,
                         n_iter=20000,
                         min_inlier_ratio=0.1,  # 这个是有几个点在规定范围内，越大越严格
                         random_state=None,
                         save_svg=False,
                         svg_dir="figuressss",
                         svg_name=None,
                         verbose=True
                         ):
    """
    RANSAC 圆拟合，用于估计 IQ 星座图的直流中心。

    参数
    ----------
    z : np.ndarray
        一维复数序列（单个通道的 IQ 数据）。
    n_iter : int
        RANSAC 最大迭代次数。
    min_inlier_ratio : float
        最小内点比例，低于该值视为拟合失败。
    random_state : int or None
        随机种子。为 None 时每次结果不同（原行为），复现实验时请显式指定。
    save_svg : bool
        是否把星座图 + 拟合圆保存为 SVG。默认 False，
        纯算法调用不再产生文件副作用。
    svg_dir : str
        启用 save_svg 时的输出目录。
    svg_name : str or None
        启用 save_svg 时的文件名。默认按时间戳命名，避免多通道互相覆盖。
    verbose : bool
        是否打印内点比例等信息。

    返回
    ----------
    (xc, yc, R) : tuple
        拟合圆心与半径；拟合失败时返回 (None, None, None)。
    """
    if verbose:
        print(f"RANSAC 圆拟合开始(共{len(z)}点)")

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
        if verbose:
            print(f"拟合失败：内点比例 {len(best_inliers) / N:.2f} < 最小要求 {min_inlier_ratio}，跳过该通道")
        return None, None, None  # 返回空值，程序不崩溃
    elif verbose:
        print(f"拟合成功：内点比例 {len(best_inliers) / N:.2f} > 最小要求 {min_inlier_ratio}")

    # refine using all inliers
    xc, yc, R = fit_circle_least_squares(best_inliers)

    if save_svg:
        _save_iq_svg(pts, xc, yc, R, svg_dir=svg_dir, svg_name=svg_name)

    return xc, yc, R


def _save_iq_svg(pts, xc, yc, R, svg_dir="figuressss", svg_name=None):
    """
    把 IQ 散点与拟合圆画成 SVG 落盘。

    注意：这是可选的副作用，已从 fit_circle_ransac_iq 主流程中剥离，
    需要显式 save_svg=True 才会执行，避免批量处理时产生大量无用文件。
    """
    if svg_name is None:
        svg_name = f"iq_circle_{int(time.time() * 1000)}.svg"

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
    ax.text(xc, yc, f"({xc:.3f}, {yc:.3f})", ha='left', va='bottom')

    os.makedirs(svg_dir, exist_ok=True)
    svg_path = os.path.join(svg_dir, svg_name)
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SVG: {svg_path}")
