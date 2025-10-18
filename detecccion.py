import cv2 as cv
import numpy as np

def _preproc(gray):
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    return cv.GaussianBlur(g, (3,3), 0)

def find_chessboard_robusto(gray, pattern_candidates=((9,6),(6,9))):
    h0, w0 = gray.shape[:2]
    for cols, rows in pattern_candidates:
        for s in (1.0, 0.75):
            g = gray if s==1.0 else cv.resize(gray,(int(w0*s),int(h0*s)), cv.INTER_AREA)
            for use_pre in (False, True):
                img = _preproc(g) if use_pre else g
                # SB (robusto)
                try:
                    flags_sb = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
                    ok, corners = cv.findChessboardCornersSB(img, (cols,rows), flags=flags_sb)
                    if ok:
                        return True, (corners/s if s!=1.0 else corners), (cols,rows)
                except Exception:
                    pass
                # clásico
                flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
                         cv.CALIB_CB_NORMALIZE_IMAGE |
                         cv.CALIB_CB_FILTER_QUADS)
                ok, corners = cv.findChessboardCorners(img,(cols,rows),flags)
                if ok:
                    term=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,1e-4)
                    corners = cv.cornerSubPix(img, corners, (11,11), (-1,-1), term)
                    return True, (corners/s if s!=1.0 else corners), (cols,rows)
    return False, None, None
