import cv2
import numpy as np
from typing import Tuple, Optional

# Check if OpenCL is available and print status
OPENCL_AVAILABLE = cv2.ocl.haveOpenCL()
if OPENCL_AVAILABLE:
    cv2.ocl.setUseOpenCL(True)

class LiveHomographyRefiner:
    """
    A class to refine homography in real-time using GPU-accelerated feature detection.
    """
    def __init__(self, n_features: int = 1500, match_percent: float = 0.9):
        """
        Initializes the ORB detector and BFMatcher for real-time use.
        
        Args:
            n_features: Max features to detect. Adjusted for a balance of speed and robustness.
            match_percent: Percentage of top matches to use.
        """
        self.n_features = n_features
        self.match_percent = match_percent
        
        # Use a balanced ORB configuration
        self.orb = cv2.ORB_create(nfeatures=self.n_features, scoreType=cv2.ORB_FAST_SCORE, fastThreshold=20)
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def find_refinement_homography(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Calculates a homography matrix to align img1 to img2.
        
        Args:
            img1: The source image (that will be warped).
            img2: The destination image (the reference).
            
        Returns:
            A tuple of (the refined homography matrix, status message).
        """
        if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
            return None, "NO_IMG"

        # Convert images to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        if OPENCL_AVAILABLE:
            gray1_umat = cv2.UMat(gray1)
            gray2_umat = cv2.UMat(gray2)
            kp1, des1_umat = self.orb.detectAndCompute(gray1_umat, None)
            kp2, des2_umat = self.orb.detectAndCompute(gray2_umat, None)
        else:
            kp1, des1_umat = self.orb.detectAndCompute(gray1, None)
            kp2, des2_umat = self.orb.detectAndCompute(gray2, None)

        if des1_umat is None or des2_umat is None or len(kp1) < 10 or len(kp2) < 10:
            return None, "NO_FEAT"

        # BFMatcher does not support UMat descriptors, so download them.
        # CRITICAL: Check for None *before* calling .get()
        des1 = des1_umat.get() if OPENCL_AVAILABLE and des1_umat is not None else des1_umat
        des2 = des2_umat.get() if OPENCL_AVAILABLE and des2_umat is not None else des2_umat
            
        if des1 is None or des2 is None:
            return None, "NO_DESC"

        try:
            matches = self.bf_matcher.match(des1, des2)
        except cv2.error:
            return None, "MATCH_ERR"

        matches = sorted(matches, key=lambda x: x.distance)
        
        num_good_matches = int(len(matches) * self.match_percent)
        good_matches = matches[:num_good_matches]
        
        if len(good_matches) < 10:
            return None, "LOW_MATCH"
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.drawIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if homography is None:
            return None, "H_FAIL"

        return homography, "OK" 