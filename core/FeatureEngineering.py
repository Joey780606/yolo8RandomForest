import numpy as np

# Keypoint indices: 0=leftEye, 1=rightEye, 2=nose, 3=leftMouth, 4=rightMouth
KeypointNames = ["LeftEye", "RightEye", "Nose", "LeftMouth", "RightMouth"]

# All 10 pairwise combinations (i, j) where i < j
PairwiseIndices = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4),
]


def computeDistance(pointA, pointB):
    """Euclidean distance between two 2D points."""
    return np.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)


def extractFeatures(keypoints, bbox):
    """
    Extract 25 normalized features from 5 facial keypoints and bounding box.

    Args:
        keypoints: Array of shape (5, 2) with (x, y) for each keypoint.
                   Order: leftEye, rightEye, nose, leftMouth, rightMouth.
        bbox: Tuple (x1, y1, x2, y2) bounding box coordinates.

    Returns:
        List of 25 float features, or None if input is invalid.
    """
    keypoints = np.array(keypoints, dtype=np.float64)
    if keypoints.shape != (5, 2):
        return None

    x1, y1, x2, y2 = bbox
    bboxWidth = x2 - x1
    bboxHeight = y2 - y1
    if bboxWidth <= 0 or bboxHeight <= 0:
        return None

    bboxDiagonal = np.sqrt(bboxWidth ** 2 + bboxHeight ** 2)
    if bboxDiagonal < 1e-6:
        return None

    # 10 normalized pairwise distances
    distances = []
    for i, j in PairwiseIndices:
        dist = computeDistance(keypoints[i], keypoints[j])
        distances.append(dist / bboxDiagonal)

    # 10 relative positions (x, y of each keypoint relative to bbox)
    positions = []
    for kp in keypoints:
        relX = (kp[0] - x1) / bboxWidth
        relY = (kp[1] - y1) / bboxHeight
        positions.append(relX)
        positions.append(relY)

    # 5 key ratios
    leftEye, rightEye, nose, leftMouth, rightMouth = keypoints

    interEyeDist = computeDistance(leftEye, rightEye)
    noseMouthMid = (leftMouth + rightMouth) / 2.0
    noseMouthDist = computeDistance(nose, noseMouthMid)
    ratioInterEyeToNoseMouth = interEyeDist / max(noseMouthDist, 1e-6)

    mouthWidth = computeDistance(leftMouth, rightMouth)
    ratioEyeWidthToMouthWidth = interEyeDist / max(mouthWidth, 1e-6)

    # Nose asymmetry: horizontal offset of nose from midpoint of eyes
    eyeMidX = (leftEye[0] + rightEye[0]) / 2.0
    noseAsymmetry = (nose[0] - eyeMidX) / max(interEyeDist, 1e-6)

    # Upper face (eyes to nose) vs lower face (nose to mouth)
    eyeMid = (leftEye + rightEye) / 2.0
    upperFaceDist = computeDistance(eyeMid, nose)
    lowerFaceDist = noseMouthDist
    ratioUpperToLower = upperFaceDist / max(lowerFaceDist, 1e-6)

    # Left side vs right side distances
    leftSideDist = computeDistance(leftEye, leftMouth)
    rightSideDist = computeDistance(rightEye, rightMouth)
    ratioLeftToRight = leftSideDist / max(rightSideDist, 1e-6)

    ratios = [
        ratioInterEyeToNoseMouth,
        ratioEyeWidthToMouthWidth,
        noseAsymmetry,
        ratioUpperToLower,
        ratioLeftToRight,
    ]

    features = distances + positions + ratios
    return features
