import numpy as np
import cv2

def get_interest_points(image, feature_width):
    """ Returns a set of interest points for the input image
    Args:
        image - can be grayscale or color, your choice.
        feature_width - in pixels, is the local feature width. It might be
            useful in this function in order to (a) suppress boundary interest
            points (where a feature wouldn't fit entirely in the image)
            or (b) scale the image filters being used. Or you can ignore it.
    Returns:
        x and y: nx1 vectors of x and y coordinates of interest points.
        confidence: an nx1 vector indicating the strength of the interest
            point. You might use this later or not.
        scale and orientation: are nx1 vectors indicating the scale and
            orientation of each interest point. These are OPTIONAL. By default you
            do not need to make scale and orientation invariant local features. 
    """
    h, w = image.shape[:2]
    
    # Placeholder that you can delete -- these are just random points
    # x = np.ceil(np.random.rand(500, 1) * w)
    # y = np.ceil(np.random.rand(500, 1) * h)
    
    # detect harris concers 
    #image = np.float32(image)
    gray = image
    
    blockSize = feature_width//2
    dst = cv2.cornerHarris(gray,2,3,0.04)

    ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)
    dst = np.uint8(dst)
    
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    centroids = np.delete(centroids,0,0) # Delete background centroid
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    test = []
    for i in range(corners.shape[0]):
        if 8<corners[i,0]<(image.shape[1]-8) and 8<corners[i,1]<(image.shape[0]-8):
            test.append(corners[i,:])
    corners =np.array(test)
    
    num_corner = corners.shape[0]
    x = corners[:,0].reshape((num_corner,1))
    y = corners[:,1].reshape((num_corner,1))


    # If you do not use (confidence, scale, orientation), just delete
    # return x, y, confidence, scale, orientation
    return x, y


def get_features(image, x, y, feature_width):
    """ Returns a set of feature descriptors for a given set of interest points. 
    Args:
        image - can be grayscale or color, your choice.
        x and y: nx1 vectors of x and y coordinates of interest points.
            The local features should be centered at x and y.
        feature_width - in pixels, is the local feature width. You can assume
            that feature_width will be a multiple of 4 (i.e. every cell of your
            local SIFT-like feature will have an integer width and height).
        If you want to detect and describe features at multiple scales or
            particular orientations you can add other input arguments.
    Returns:
        features: the array of computed features. It should have the
            following size: [length(x) x feature dimensionality] (e.g. 128 for
            standard SIFT)
    """
    # Placeholder that you can delete. Empty features.
    features = np.zeros((x.shape[0], 128))
    

    
    # convolve a Gaussian kernel
    blur = cv2.GaussianBlur(image,(5,5),1)
    # Calculate the image gradient
    dx = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=5)
    dy = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=5)
    # Calculate the magnitude and orientation of each pixel
    mag = np.sqrt(dx**2+dy**2)
    eps = 0.00001
    orient = np.arctan(dy/(dx+0.00001))
    
    num_bins = 8
    p = np.pi 
    
    for k in range(x.shape[0]):
    
        features_window = np.zeros((128,1))
        current_point_x = np.round(x[k]).astype('int')
        current_point_y = np.round(y[k]).astype('int')
        
        x_start = int(current_point_x - (feature_width // 2))
        y_start = int(current_point_y - (feature_width // 2))
        x_end = int(current_point_x + (feature_width // 2))
        y_end = int(current_point_y + (feature_width // 2))
                  
        # For each detected point, consider a 16*16 window centered at this point
        mag_window = mag[y_start:y_end, x_start:x_end]
        kernel_1d  = cv2.getGaussianKernel(ksize = feature_width, sigma = feature_width//2)
        kernel_2d = kernel_1d * kernel_1d.T
        mag_window = mag_window * kernel_2d
        orient_window = orient[y_start:y_end, x_start:x_end]
 
        # Consider each 4*4 cell of the window
        x_initial = 0
        y_initial = 0
        index = 0
        for i in range(0,4):
            for j in range(0,4):
                x_start_cell = x_initial
                x_end_cell = x_start_cell + 4
                y_start_cell = y_initial
                y_end_cell = y_start_cell + 4
                mag_cell = mag_window[y_start_cell:y_end_cell, x_start_cell:x_end_cell]
                orient_cell = orient_window[y_start_cell:y_end_cell, x_start_cell:x_end_cell]

                # Cast the orientations of each pixel into 8 bins
                cell_dir = np.zeros((8,1))
                
                
                for px in range(mag_cell.shape[1]):
                    for py in range(mag_cell.shape[0]):
                        mag_pixel = mag_cell[py,px]
                        t = orient_cell[py,px]

                        if (t >= (p/2*(-4/4))) and (t <= (p/2*(-3/4))):
                            cell_dir[0] += mag_pixel
                        if (t > (p/2*(-3/4))) and (t <= (p/2*(-2/4))):
                            cell_dir[1] += mag_pixel
                        if (t > (p/2*(-2/4))) and (t <= (p/2*(-1/4))):
                            cell_dir[2] += mag_pixel
                        if (t > (p/2*(-1/4))) and (t <= 0):
                            cell_dir[3] += mag_pixel
                        if (t > 0) and (t <= (p/2*(1/4))):
                            cell_dir[4] += mag_pixel
                        if (t > (p/2*(1/4))) and (t <= (p/2*(2/4))):
                            cell_dir[5] += mag_pixel
                        if (t > (p/2*(2/4))) and (t <= (p/2*(3/4))):
                            cell_dir[6] += mag_pixel
                        if (t > (p/2*(3/4))) and (t <= (p/2*(4/4))):
                            cell_dir[7] += mag_pixel
                   


                features_window[(index*8):(index*8+8)] = cell_dir 
                index += 1
                y_initial = y_start_cell + 4

            x_initial = x_start_cell + 4
            y_initial = 0

        features_vector = (features_window/np.linalg.norm(features_window)).reshape((128,1))
        features[k,:] = features_vector.T

   
    
    return features


def match_features(features1, features2, threshold=0.0):
    """ 
    Args:
        features1 and features2: the n x feature dimensionality features
            from the two images.
        threshold: a threshold value to decide what is a good match. This value 
            needs to be tuned.
        If you want to include geometric verification in this stage, you can add
            the x and y locations of the features as additional inputs.
    Returns:
        matches: a k x 2 matrix, where k is the number of matches. The first
            column is an index in features1, the second column is an index
            in features2. 
        Confidences: a k x 1 matrix with a real valued confidence for every
            match.
        matches' and 'confidences' can be empty, e.g. 0x2 and 0x1.
    """
    
    # Placeholder that you can delete. Random matches and confidences
    num_features = min(features1.shape[0], features2.shape[0])
    matched = np.zeros((num_features, 2))
    matched[:, 0] = np.random.permutation(num_features)
    matched[:, 1] = np.random.permutation(num_features)
    confidence = np.random.rand(num_features, 1)
    
    
    # Match each key-point in features1 and calculate confidence value

    
    for i in range(num_features):
        point1 = features1[i,:]
        distance_list = np.zeros((num_features,1))
        for j in range(num_features):
            point2 = features2[j,:]
            distance = np.linalg.norm(point1-point2)
            distance_list[j] = distance 

        min_index = np.argmin(distance_list)

        distance_list_order = np.sort(distance_list,axis=0)
        match_top = distance_list_order[0]
        match_second = distance_list_order[1]
        np.seterr(invalid='ignore')
        ratio = match_top/match_second
        if ratio < 0.75:
            conf = 1/ratio
            confidence[i] = conf
            matched[i,0] = i
            matched[i,1] = min_index
    
    '''
    features1 = np.float32(features1)
    features2 = np.float32(features2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(features1,features2,k=2)
    
    kk = 0
    for m,n in matches:
        
        if m.distance < 0.99*n.distance:
            match1 = m.queryIdx
            match2 = m.trainIdx
            matched[kk,0] = match1
            matched[kk,1] = match2
            ratio = m.distance/n.distance
            confidence[kk] = 1/ratio
            kk = kk+1
     '''
    
    # Sort the matches so that the most confident onces are at the top of the
    # list. You should probably not delete this, so that the evaluation
    # functions can be run on the top matches easily.
    order = np.argsort(confidence, axis=0)[::-1, 0]
    confidence = confidence[order, :]
    matched = matched[order, :]


    return matched, confidence