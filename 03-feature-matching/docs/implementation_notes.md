# Implementation Notes - Assignment 3

## Harris Corner Detection Details

### Algorithm Steps

1. **Compute Image Derivatives:**
```python
   Ix = cv2.Sobel(image, CV_32F, 1, 0, ksize=3)
   Iy = cv2.Sobel(image, CV_32F, 0, 1, ksize=3)
```

2. **Compute Structure Tensor:**
```python
   Ixx = Ix * Ix
   Iyy = Iy * Iy
   Ixy = Ix * Iy
   
   # Apply Gaussian smoothing
   Sxx = cv2.GaussianBlur(Ixx, (k, k), sigma)
   Syy = cv2.GaussianBlur(Iyy, (k, k), sigma)
   Sxy = cv2.GaussianBlur(Ixy, (k, k), sigma)
```

3. **Compute Harris Response:**
```python
   det_M = Sxx * Syy - Sxy * Sxy
   trace_M = Sxx + Syy
   R = det_M - k * (trace_M**2)
```

4. **Non-Maximum Suppression:**
```python
   corners = (R > threshold) & (R == local_max)
```

### Parameter Effects

**blockSize:**
- Larger → Fewer, more stable corners
- Smaller → More corners, but noisier
- Optimal: 4 for most cases

**Threshold (m × R.max()):**
- Higher m → Only strongest corners
- Lower m → More corners, including weaker ones
- Optimal: 0.002 for balanced performance

---

## SIFT Descriptor Implementation

### Window Size Selection

**Why 16×16?**
- Large enough to be distinctive
- Small enough to be repeatable
- Multiple of 4 for 4×4 cell division
- Covers ~2-3 pixel radius around feature

### Cell Size (4×4 pixels)

**Why 4×4 cells?**
- 16 cells total provides spatial layout
- 4 pixels per cell balances detail vs. robustness
- Results in manageable 128-D vector (16×8)

### Orientation Bins

**Why 8 bins?**
- 45° per bin covers gradient space well
- More bins → higher dimensionality
- Fewer bins → loss of discriminative power
- 8 is empirically optimal (Lowe 2004)

### Gaussian Weighting

**Purpose:**
- Down-weight features far from center
- Reduces edge effects
- Makes descriptor more localized

**Parameters:**
```python
sigma = feature_width / 2 = 8
kernel_size = feature_width = 16
```

---

## Ratio Test Matching

### Why It Works

**Problem with nearest neighbor:**
```
Feature A → [Feature B: dist=10, Feature C: dist=11]
  ↑
Both B and C are similar distances → ambiguous!
```

**Ratio test solution:**
```
ratio = 10/11 = 0.91 → High ratio → Reject (ambiguous)

Feature A → [Feature B: dist=5, Feature C: dist=15]
ratio = 5/15 = 0.33 → Low ratio → Accept (clear winner)
```

### Threshold Selection

| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 0.6 | High | Low | Medium |
| 0.7 | High | Medium | Good |
| 0.8 | Medium | High | **Best** |
| 0.9 | Low | High | Poor |

**Lowe's original: 0.8** - Best trade-off

---

## Failure Cases Analysis

### Episcopal Gaudi (10% accuracy)

**Why so low?**

1. **Large Viewpoint Change (60°+)**
   - Descriptor appearance changes significantly
   - Foreshortening distorts local structure
   
2. **Scale Change**
   - No scale normalization
   - Features at different scales don't match

3. **Repetitive Architecture**
   - Many similar-looking windows and arches
   - Ambiguous matches even with ratio test

4. **Limited Overlap**
   - Different parts of building visible
   - Few truly corresponding features

**What would help:**
- Scale-space detection (DOG pyramid)
- Rotation normalization
- Affine adaptation
- Deep learned descriptors

---

## Code Structure

### match_functions.py
```python
def get_interest_points(image):
    """
    25 points - Harris corner detection
    
    Returns:
        x: (N,) array of x-coordinates
        y: (N,) array of y-coordinates
    """
    # Implementation here
    pass

def get_features(image, x, y, feature_width=16):
    """
    40 points - SIFT-like descriptor extraction
    
    Returns:
        features: (N, 128) array of descriptors
    """
    # Implementation here
    pass

def match_features(features1, features2):
    """
    15 points - Ratio test matching
    
    Returns:
        matches: (M, 2) array of matched indices
        confidences: (M,) array of confidence scores
    """
    # Implementation here
    pass
```

---

## Lessons Learned

### 1. Parameter Tuning is Critical

Small changes in blockSize or threshold dramatically affect results.

### 2. Trade-offs Everywhere

- More corners → More computation, more potential matches
- Larger descriptors → More distinctive, but slower
- Stricter ratio test → Fewer false positives, but miss true matches

### 3. No Silver Bullet

Different scenes require different parameters. Episcopal Gaudi shows limitations of handcrafted features.

### 4. Evaluation Matters

Ground truth correspondences allow quantitative evaluation. Without them, qualitative assessment is subjective.

---

## Computational Complexity

### Harris Corners
- Image gradients: O(WHk²)
- Structure tensor: O(WHk²)  
- Total: O(WH) per image

### SIFT Descriptors
- Per keypoint: O(n²) where n=16
- For N keypoints: O(Nn²)

### Matching
- Distance computation: O(N₁N₂D) where D=128
- Sorting: O(N₁N₂log(N₂))
- Total: O(N₁N₂)

**For typical image:**
- W×H = 500×750
- N = 1000 keypoints
- Total time: ~1 second (CPU)

---

## Future Improvements

### Short-term
- [ ] Optimize Harris with separable filters
- [ ] Parallelize descriptor computation
- [ ] Use KD-tree for faster NN search

### Medium-term
- [ ] Implement scale-space detection
- [ ] Add rotation normalization
- [ ] RANSAC geometric verification

### Long-term
- [ ] Deep learning descriptors (SuperPoint)
- [ ] End-to-end learned matching (SuperGlue)
- [ ] Real-time performance (GPU)