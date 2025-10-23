import os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def calc_disparidad(
    rectified_dir="data/rectified",
    data_dir="data",
    method="SGBM",
    min_disparity=0,
    num_disparities=160,  # aumentado para más rango
    block_size=7,         # aumentado para reducir ruido
    use_wls=False,
    pair_index=None,
    name_pattern=None,
    use_valid_roi=True,
    n_show=5,
    random_state=None
):
    """
    Calcula mapas de disparidad con filtrado robusto de outliers.
    """
    # --- pares rectificados ---
    lefts  = sorted(glob.glob(os.path.join(rectified_dir, "*left*_rect.png")))
    rights = sorted(glob.glob(os.path.join(rectified_dir, "*right*_rect.png")))
    assert len(lefts) == len(rights) > 0, "No se encontraron pares rectificados."
    pares = list(zip(lefts, rights))

    # --- filtrar por patrón / índice ---
    if name_pattern:
        pares = [(l, r) for (l, r) in pares if name_pattern in os.path.basename(l)]
        assert len(pares) > 0, f"No encontré pares que contengan '{name_pattern}'."
    if pair_index is not None:
        assert 0 <= pair_index < len(pares), "pair_index fuera de rango."
        pares = [pares[pair_index]]

    # --- (opcional) cargar ROIs válidas ---
    roi1 = roi2 = None
    if use_valid_roi:
        pkl_maps = os.path.join(data_dir, "stereo_maps.pkl")
        if os.path.exists(pkl_maps):
            try:
                with open(pkl_maps, "rb") as f:
                    maps = pickle.load(f)
                roi1 = maps.get("validRoi1")
                roi2 = maps.get("validRoi2")
            except Exception:
                pass

    # --- elegir qué índices mostrar ---
    rng = np.random.default_rng(random_state)
    k = min(n_show, len(pares))
    show_indices = set(rng.choice(len(pares), size=k, replace=False).tolist())

    # --- preprocesado ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    def prep(gray):
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (3,3), 0)
        return g

    # --- matcher ---
    if method.upper() == "BM":
        stereo = cv2.StereoBM_create(numDisparities=num_disparities,
                                     blockSize=max(9, block_size|1))
    else:
        bs = max(5, block_size | 1)  # asegurar impar y mínimo 5
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=bs,
            P1=8 * (bs ** 2),
            P2=32 * (bs ** 2),
            disp12MaxDiff=1,
            uniquenessRatio=10,        # reducido para más matches
            speckleWindowSize=100,     # reducido para objetos pequeños
            speckleRange=32,           # aumentado para más tolerancia
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    # --- WLS opcional ---
    have_wls = False
    if use_wls and method.upper() != "BM":
        try:
            import cv2.ximgproc as xip
            wls = xip.createDisparityWLSFilter(stereo)
            right_matcher = xip.createRightMatcher(stereo)
            have_wls = True
        except Exception:
            print("⚠️ No se pudo importar cv2.ximgproc; continuo sin WLS.")

    # --- procesar pares ---
    for i, (lp, rp) in enumerate(pares):
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)
        if Lc is None or Rc is None:
            continue
        L = cv2.cvtColor(Lc, cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(Rc, cv2.COLOR_BGR2GRAY)

        # recorte ROI
        if roi1 and roi2:
            x1,y1,w1,h1 = roi1; x2,y2,w2,h2 = roi2
            x  = max(x1, x2);  y  = max(y1, y2)
            xe = min(x1+w1, x2+w2); ye = min(y1+h1, y2+h2)
            if xe > x and ye > y:
                Lc, Rc = Lc[y:ye, x:xe], Rc[y:ye, x:xe]
                L,  R  =  L[y:ye,  x:xe],  R[y:ye,  x:xe]

        # preprocesado
        Lg, Rg = prep(L), prep(R)

        # disparidad
        if have_wls:
            dispL = stereo.compute(Lg, Rg).astype(np.float32) / 16.0
            dispR = right_matcher.compute(Rg, Lg).astype(np.float32) / 16.0
            wls.setLambda(8000.0); wls.setSigmaColor(1.0)
            disp = wls.filter(np.int16(dispL*16), Lc, disparity_map_right=np.int16(dispR*16))
            disp = disp.astype(np.float32) / 16.0
        else:
            disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0

        disp = cv2.medianBlur(disp, 5)

        # *** FILTRADO DE OUTLIERS ***
        # Aplicar máscara: disp > umbral mínimo (más agresivo para evitar Z > 2m)
        disp_filtered = disp.copy()
        disp_filtered[disp < 5.0] = 0  # disparidad mínima para objetos cercanos (evita outliers lejanos)

        # visualización con percentiles
        dmask = disp_filtered > 0
        if dmask.any():
            p5, p95 = np.percentile(disp_filtered[dmask], [5, 95])
        else:
            p5, p95 = 0, np.max(disp_filtered)
        disp_vis = np.clip((disp_filtered - p5) / max(p95 - p5, 1e-6), 0, 1)
        disp_u8 = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

        # mostrar solo algunos
        if i in show_indices:
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1); plt.imshow(L, cmap="gray"); plt.title("Left rect");  plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(R, cmap="gray"); plt.title("Right rect"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)); plt.title("Disparidad"); plt.axis("off")
            plt.tight_layout()
            plt.show()