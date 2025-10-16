import os, glob, pickle, cv2, random
import numpy as np
import matplotlib.pyplot as plt

def rectificar_imagenes(
    data_dir="data",
    captures_dir="data/captures",
    out_dir="data/rectified",
    visualizar=True,
    n_muestras=5,
    n_lineas=8,
):
    """
    Rectifica pares estéreo (left/right) usando mapas en stereo_maps.pkl.
    Opcionalmente, selecciona 'n_muestras' pares al azar y dibuja líneas epipolares
    horizontales para verificar la alineación (mismo 'y' en left y right).

    Args:
        data_dir: carpeta con 'stereo_maps.pkl' (contiene left_map/right_map y Q).
        captures_dir: carpeta con imágenes originales 'left_*.jpg' y 'right_*.jpg'.
        out_dir: carpeta donde se guardan las imágenes rectificadas.
        visualizar: si True, muestra pares con líneas epipolares.
        n_muestras: cantidad de pares que se muestran (aleatorios).
        n_lineas: cantidad de líneas horizontales a dibujar en cada par.

    Returns:
        Q (np.ndarray or None): matriz de reproyección 3D si está en el pkl.
        pares_guardados (list[tuple[str,str]]): lista de rutas (left_rect, right_rect).
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Cargar mapas ---
    with open(os.path.join(data_dir, "stereo_maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    map1x, map1y = maps["left_map_x"],  maps["left_map_y"]
    map2x, map2y = maps["right_map_x"], maps["right_map_y"]
    Q = maps.get("Q", None)

    # --- Buscar pares originales ---
    lefts  = sorted(glob.glob(os.path.join(captures_dir, "left_*.jpg")))
    rights = sorted(glob.glob(os.path.join(captures_dir, "right_*.jpg")))
    assert len(lefts) == len(rights) and len(lefts) > 0, "No se encontraron pares left/right."

    # --- Rectificar y guardar ---
    pares_guardados = []
    for i, (lp, rp) in enumerate(zip(lefts, rights), start=1):
        L = cv2.imread(lp, cv2.IMREAD_COLOR)
        R = cv2.imread(rp, cv2.IMREAD_COLOR)
        if L is None or R is None:
            continue

        rectL = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

        baseL = os.path.splitext(os.path.basename(lp))[0]
        baseR = os.path.splitext(os.path.basename(rp))[0]
        outL  = os.path.join(out_dir, f"{i:03d}_{baseL}_rect.png")
        outR  = os.path.join(out_dir, f"{i:03d}_{baseR}_rect.png")
        cv2.imwrite(outL, rectL)
        cv2.imwrite(outR, rectR)
        pares_guardados.append((outL, outR))

    # --- Visualizar N pares con líneas epipolares ---
    if visualizar and len(pares_guardados) > 0:
        muestras = random.sample(pares_guardados, k=min(n_muestras, len(pares_guardados)))
        for outL, outR in muestras:
            Lr = cv2.imread(outL, cv2.IMREAD_COLOR)
            Rr = cv2.imread(outR, cv2.IMREAD_COLOR)
            if Lr is None or Rr is None:
                continue

            # Convertir a RGB para matplotlib
            Lrgb = cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB)
            Rrgb = cv2.cvtColor(Rr, cv2.COLOR_BGR2RGB)

            h, w = Lrgb.shape[:2]
            ys = np.linspace(40, h-40, n_lineas, dtype=int)  # evita bordes

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(Lrgb); axes[0].set_title("Left rect");  axes[0].axis("off")
            axes[1].imshow(Rrgb); axes[1].set_title("Right rect"); axes[1].axis("off")

            for y in ys:
                # mismas 'y' en ambas imágenes (condición epipolar tras rectificación)
                axes[0].plot([0, w-1], [y, y], color="lime", linewidth=1)
                axes[1].plot([0, w-1], [y, y], color="lime", linewidth=1)

            plt.tight_layout()
            plt.show()

    return Q, pares_guardados
