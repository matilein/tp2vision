import os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def calc_disparidad(
    rectified_dir="data/rectified",
    data_dir="data",              # para leer validRoi (opcional)
    method="SGBM",
    min_disparity=0,
    num_disparities=224,          # múltiplo de 16
    block_size=9,                 # 7–11 suele ir bien
    use_wls=False,                # requiere opencv-contrib-python
    pair_index=None,
    name_pattern=None,
    use_valid_roi=True,           # recorta a intersección de ROIs válidas si existen
    n_show=5,                     # <<< mostrar solo 5 al azar
    random_state=None             # <<< semilla para reproducibilidad
):
    """
    Calcula mapas de disparidad para TODOS los pares seleccionados,
    pero SOLO muestra n_show casos aleatorios.

    - Preprocesado CLAHE + blur
    - SGBM (o BM) con parámetros robustos
    - WLS opcional (si está cv2.ximgproc)
    - Filtro por índice o patrón de nombre
    - Recorte opcional por ROI válida (validRoi1/2) si existe en stereo_maps.pkl
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

    # --- elegir qué índices se van a mostrar (pero procesamos TODOS) ---
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
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=max(5, block_size|1),
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
    if use_wls and method.upper() != "BM":
        try:
            import cv2.ximgproc as xip
            wls = xip.createDisparityWLSFilter(stereo)
            right_matcher = xip.createRightMatcher(stereo)
            have_wls = True
        except Exception:
            print("⚠️ No se pudo importar cv2.ximgproc; continuo sin WLS.")

    # --- procesar pares (se muestran solo algunos) ---
    for i, (lp, rp) in enumerate(pares):
        # leo color para guiar WLS y mostrar; y grises para el matcher
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)
        if Lc is None or Rc is None:
            continue
        L = cv2.cvtColor(Lc, cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(Rc, cv2.COLOR_BGR2GRAY)

        # recorte a intersección de ROIs válidas (si existen)
        if roi1 and roi2:
            x1,y1,w1,h1 = roi1; x2,y2,w2,h2 = roi2
            x  = max(x1, x2);  y  = max(y1, y2)
            xe = min(x1+w1, x2+w2); ye = min(y1+h1, y2+h2)
            if xe > x and ye > y:
                Lc, Rc = Lc[y:ye, x:xe], Rc[y:ye, x:xe]
                L,  R  =  L[y:ye,  x:xe],  R[y:ye,  x:xe]

        # preprocesado
        Lg, Rg = prep(L), prep(R)

        # disparidad (con/sin WLS)
        if have_wls:
            dispL = stereo.compute(Lg, Rg).astype(np.float32) / 16.0
            dispR = right_matcher.compute(Rg, Lg).astype(np.float32) / 16.0
            wls.setLambda(8000.0); wls.setSigmaColor(1.0)
            disp = wls.filter(np.int16(dispL*16), Lc, disparity_map_right=np.int16(dispR*16))
            disp = disp.astype(np.float32) / 16.0
        else:
            disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0

        # suavizado
        disp = cv2.medianBlur(disp, 5)

        # visualización con percentiles (evita saturación)
        dmask = np.isfinite(disp)
        if dmask.any():
            p5, p95 = np.percentile(disp[dmask], [5, 95])
        else:
            p5, p95 = np.min(disp), np.max(disp)
        disp_vis = np.clip((disp - p5) / max(p95 - p5, 1e-6), 0, 1)
        disp_u8 = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

        # --- mostrar solo si este índice fue elegido ---
        if i in show_indices:
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1); plt.imshow(L, cmap="gray"); plt.title("Left rect");  plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(R, cmap="gray"); plt.title("Right rect"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)); plt.title("Disparidad"); plt.axis("off")
            plt.tight_layout()
            plt.show()
