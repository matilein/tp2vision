import os, glob, pickle, cv2, numpy as np
import matplotlib.pyplot as plt

def calc_profundidad(
    data_dir="data",
    rectified_dir="data/rectified",
    out_dir="data/pointcloud",
    min_disparity=0,
    num_disparities=224,
    block_size=9,
    z_min_m=0.1,     # ← ajustado
    z_max_m=2.0,
    use_wls=False,
    pair_index=None,
    name_pattern=None,
    n_show=5,
    random_state=None
):
    """
    Calcula profundidad con filtrado robusto de outliers en Z.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Cargar Q ---
    with open(os.path.join(data_dir, "stereo_maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    Q = maps["Q"]
    roi1 = maps.get("validRoi1")
    roi2 = maps.get("validRoi2")

    # --- Seleccionar pares ---
    lefts  = sorted(glob.glob(os.path.join(rectified_dir, "*left*_rect.png")))
    rights = sorted(glob.glob(os.path.join(rectified_dir, "*right*_rect.png")))
    assert len(lefts) == len(rights) > 0, "No hay pares rectificados."
    pares = list(zip(lefts, rights))

    if name_pattern:
        pares = [(l, r) for (l, r) in pares if name_pattern in os.path.basename(l)]
    if pair_index is not None:
        pares = [pares[pair_index]]

    rng = np.random.default_rng(random_state)
    k = min(n_show, len(pares))
    show_indices = set(rng.choice(len(pares), size=k, replace=False).tolist())

    # --- Preprocesado ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    def prep(img_gray):
        g = clahe.apply(img_gray)
        return cv2.GaussianBlur(g, (3, 3), 0)

    # --- SGBM ---
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * (block_size ** 2),
        P2=32 * (block_size ** 2),
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=200,
        speckleRange=16,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # --- WLS opcional ---
    have_wls = False
    if use_wls:
        try:
            import cv2.ximgproc as xip
            wls = xip.createDisparityWLSFilter(stereo)
            right_matcher = xip.createRightMatcher(stereo)
            have_wls = True
        except Exception:
            print("⚠️ Sin WLS")

    def _save_ply_simple(points_xyz, colors_bgr, path):
        pts = points_xyz.reshape(-1, 3)
        cols = colors_bgr.reshape(-1, 3)[:, ::-1]
        valid = np.isfinite(pts).all(axis=1)
        pts, cols = pts[valid], cols[valid]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    ply_paths = []
    for i, (lp, rp) in enumerate(pares):
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)
        if Lc is None or Rc is None:
            continue
        Lg = prep(cv2.cvtColor(Lc, cv2.COLOR_BGR2GRAY))
        Rg = prep(cv2.cvtColor(Rc, cv2.COLOR_BGR2GRAY))

        # ROI
        if roi1 and roi2:
            x1, y1, w1, h1 = roi1; x2, y2, w2, h2 = roi2
            x = max(x1, x2); y = max(y1, y2)
            xe = min(x1+w1, x2+w2); ye = min(y1+h1, y2+h2)
            if xe > x and ye > y:
                Lc, Rc = Lc[y:ye, x:xe], Rc[y:ye, x:xe]
                Lg, Rg = Lg[y:ye, x:xe], Rg[y:ye, x:xe]

        # Disparidad
        if have_wls:
            dispL = stereo.compute(Lg, Rg).astype(np.float32) / 16.0
            dispR = right_matcher.compute(Rg, Lg).astype(np.float32) / 16.0
            wls.setLambda(8000.0); wls.setSigmaColor(1.0)
            disp = wls.filter(np.int16(dispL*16), Lc, disparity_map_right=np.int16(dispR*16))
            disp = disp.astype(np.float32) / 16.0
        else:
            disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0

        disp = cv2.medianBlur(disp, 5)

        # *** FILTRADO CLAVE: Disparidad mínima + filtrado de Z outliers ***
        mask_disp = disp > 5.0  # disparidad mínima razonable
        pts3d = cv2.reprojectImageTo3D(disp, Q)
        pts3d[~mask_disp] = np.nan

        Z = pts3d[:,:,2]
        Z_valid = Z[np.isfinite(Z)]
        
        if len(Z_valid) > 0:
            # Percentiles robustos
            z_p5, z_p95 = np.percentile(Z_valid, [5, 95])
            z_min_robust = max(z_min_m, z_p5 - 0.2)
            z_max_robust = min(z_max_m, z_p95 + 0.5)
            
            # Máscara final: Z en rango razonable
            mask_z = (Z > z_min_robust) & (Z < z_max_robust) & np.isfinite(Z)
            mask_final = mask_disp & mask_z
            
            pts3d_filtered = pts3d[mask_final]
            colors_filtered = Lc[mask_final]
        else:
            pts3d_filtered = np.empty((0,3))
            colors_filtered = np.empty((0,3))

        # Guardar
        base = os.path.splitext(os.path.basename(lp))[0]
        ply = os.path.join(out_dir, f"{base}.ply")
        _save_ply_simple(pts3d_filtered, colors_filtered, ply)
        ply_paths.append(ply)
        print(f"✅ {i+1:03d}: {base} | {len(pts3d_filtered)} pts")

        # Visualizar algunos
        if i in show_indices:
            Zm = Z.copy()
            Zm[~mask_final] = np.nan
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].imshow(cv2.cvtColor(Lc, cv2.COLOR_BGR2RGB))
            ax[0].set_title("Left"); ax[0].axis("off")
            
            if len(Z_valid) > 0:
                im = ax[1].imshow(Zm, cmap="magma", vmin=z_min_robust, vmax=z_max_robust)
                fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            ax[1].set_title("Profundidad Z [m]"); ax[1].axis("off")
            plt.tight_layout(); plt.show()

    return ply_paths