import os, glob, pickle, cv2, numpy as np
import matplotlib.pyplot as plt

def calc_profundidad(
    data_dir="data",
    rectified_dir="data/rectified",
    out_dir="data/pointcloud",
    min_disparity=0,
    num_disparities=224,
    block_size=9,
    z_min_m=0.3,
    z_max_m=2.0,
    use_wls=False,
    pair_index=None,
    name_pattern=None,
    n_show=5,               # <<< mostrar solo 5
    random_state=None       # <<< para reproducibilidad
):
    """
    Calcula disparidad para TODOS los pares, pero SOLO muestra n_show casos aleatorios.
    Devuelve la lista de rutas .ply generadas.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Cargar Q y ROIs ---
    with open(os.path.join(data_dir, "stereo_maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    Q = maps["Q"]
    roi1 = maps.get("validRoi1")
    roi2 = maps.get("validRoi2")

    # --- Seleccionar pares rectificados ---
    lefts  = sorted(glob.glob(os.path.join(rectified_dir, "*left*_rect.png")))
    rights = sorted(glob.glob(os.path.join(rectified_dir, "*right*_rect.png")))
    assert len(lefts) == len(rights) > 0, "No hay pares rectificados."
    pares = list(zip(lefts, rights))

    if name_pattern:
        pares = [(l, r) for (l, r) in pares if name_pattern in os.path.basename(l)]
        assert len(pares) > 0, f"No encontré pares que contengan '{name_pattern}'."
    if pair_index is not None:
        assert 0 <= pair_index < len(pares), "pair_index fuera de rango."
        pares = [pares[pair_index]]

    # --- Elegir cuáles mostrar (pero procesar todos) ---
    rng = np.random.default_rng(random_state)
    k = min(n_show, len(pares))
    show_indices = set(rng.choice(len(pares), size=k, replace=False).tolist())

    # --- Preprocesado: CLAHE + blur ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    def prep(img_gray):
        g = clahe.apply(img_gray)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return g

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
            print("⚠️ No se pudo importar cv2.ximgproc; continuo sin WLS.")

    def _save_ply_simple(points_xyz, colors_bgr, path):
        pts = points_xyz.reshape(-1, 3)
        cols = colors_bgr.reshape(-1, 3)[:, ::-1]   # BGR→RGB
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
        # --- Leer y preparar ---
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)
        if Lc is None or Rc is None:
            continue
        Lg = prep(cv2.cvtColor(Lc, cv2.COLOR_BGR2GRAY))
        Rg = prep(cv2.cvtColor(Rc, cv2.COLOR_BGR2GRAY))

        # --- Recorte por ROI intersección ---
        if roi1 and roi2:
            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2
            x  = max(x1, x2)
            y  = max(y1, y2)
            xe = min(x1 + w1, x2 + w2)
            ye = min(y1 + h1, y2 + h2)
            if xe > x and ye > y:
                Lc, Rc = Lc[y:ye, x:xe], Rc[y:ye, x:xe]
                Lg, Rg = Lg[y:ye, x:xe], Rg[y:ye, x:xe]

        # --- Disparidad ---
        if have_wls:
            dispL = stereo.compute(Lg, Rg).astype(np.float32) / 16.0
            dispR = right_matcher.compute(Rg, Lg).astype(np.float32) / 16.0
            wls.setLambda(8000.0)
            wls.setSigmaColor(1.0)
            disp = wls.filter(np.int16(dispL * 16), Lc, disparity_map_right=np.int16(dispR * 16))
            disp = disp.astype(np.float32) / 16.0
        else:
            disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0

        disp = cv2.medianBlur(disp, 5)

        # --- Reproyección a 3D ---
        mask = disp > (min_disparity + 0.5)
        pts3d = cv2.reprojectImageTo3D(disp, Q)
        pts3d[~mask] = np.nan

        # --- Z en metros si hace falta ---
        Z = pts3d[..., 2]
        Zm = Z.copy()
        Zm[~np.isfinite(Zm)] = np.nan
        if np.nanmedian(Zm) > 100:   # heurística mm→m
            Zm /= 1000.0

        # --- Guardar .ply (si Z original parece mm, convertimos) ---
        pts_save = pts3d.copy()
        if np.nanmedian(Z) > 100:
            pts_save /= 1000.0
        base = os.path.splitext(os.path.basename(lp))[0].replace("_left_rect", "")
        ply = os.path.join(out_dir, f"{base}.ply")
        _save_ply_simple(pts_save, Lc, ply)
        ply_paths.append(ply)

        # --- Mostrar solo si este índice fue elegido ---
        if i in show_indices:
            p5, p95 = np.nanpercentile(Zm, [5, 95])
            vmin = max(z_min_m, p5) if np.isfinite(p5) else z_min_m
            vmax = min(z_max_m, p95) if np.isfinite(p95) else z_max_m

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].imshow(cv2.cvtColor(Lc, cv2.COLOR_BGR2RGB)); ax[0].set_title("Left rect (ROI)"); ax[0].axis("off")
            im = ax[1].imshow(Zm, cmap="magma", vmin=vmin, vmax=vmax)
            ax[1].set_title("Profundidad Z [m]"); ax[1].axis("off")
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            plt.tight_layout(); plt.show()

    return ply_paths
