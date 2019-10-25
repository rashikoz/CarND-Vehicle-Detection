from os import path
import glob
import cv2
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


class InputDataProvider(object):
    def __init__(self, file_location):
        self.non_vehicle_files_loc = path.join(file_location, 'non-vehicles')
        self.vehicle_files_loc = path.join(file_location, 'vehicles')
        self.vehicles_image_list = []
        self.non_vehicle_image_list = []
        self.train_samples = []
        self.validation_samples = []
        for filename in glob.iglob(self.vehicle_files_loc +'/**/*.png', recursive=True):
            self.vehicles_image_list.append(filename)
        for filename in glob.iglob(self.non_vehicle_files_loc +'/**/*.png', recursive=True):
            self.non_vehicle_image_list.append(filename)

    def get_num_vehicle_images(self):
        return len(self.vehicles_image_list)
    
    def get_num_non_vehicle_images(self):
        return len(self.non_vehicle_image_list)
    
    def get_vehicle_image(self, index):
        bgr_image = cv2.imread(self.vehicles_image_list[index])
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    def get_non_vehicle_image(self, index):
        bgr_image = cv2.imread(self.non_vehicle_image_list[index])        
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    def get_resized_image(self, index, type='vehicle', new_size=(64,64,3)):
        if type == 'vehicle':
            bgr_image = cv2.imread(self.vehicles_image_list[index])
        elif type == 'non-vehicle':
            bgr_image = cv2.imread(self.non_vehicle_image_list[index])
        return cv2.resize(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), new_size)

def get_histogram_features(input_image, nbins=32, bins_range=(0,256)):
    chan_1 = np.histogram(input_image[:,:,0], bins=nbins, range=bins_range)
    chan_2 = np.histogram(input_image[:,:,1], bins=nbins, range=bins_range)
    chan_3 = np.histogram(input_image[:,:,2], bins=nbins, range=bins_range)
    return chan_1, chan_2, chan_3

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = []
        for each_channel in cv2.split(img):
            features.extend(hog(each_channel, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec))
        return np.array(features)

def bin_spatial(input_image, size=(32, 32)):
    color1 = cv2.resize(input_image[:,:,0], size).ravel()
    color2 = cv2.resize(input_image[:,:,1], size).ravel()
    color3 = cv2.resize(input_image[:,:,2], size).ravel()
    return color1, color2, color3

def convert_color_space(input_image, dest_color_space ='rgb'):
    if dest_color_space == 'hsv':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    elif dest_color_space == 'luv':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2LUV)
    elif dest_color_space == 'hls':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HLS)
    elif dest_color_space == 'yuv':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
    elif dest_color_space == 'ycrcb':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
    elif dest_color_space == 'lab':
        mod_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
    else:
        mod_image = np.copy(input_image)
    return mod_image

def extract_image_feature(input_image, cspace='hls', spatial_size=(32, 32),
                          hist_bins=32, hist_range=(0, 256), orient = 9, 
                          pix_per_cell=8, cell_per_block=2):
    # convert to  required color space
    feature_image = convert_color_space(input_image, cspace)
    # Apply bin_spatial() to get spatial color features
    spatial_value = bin_spatial(feature_image, size=spatial_size)
    spatial_features = np.concatenate((spatial_value[0], 
                                       spatial_value[1],
                                       spatial_value[2]))
    # Apply color_hist() also with a color space option now
    hist_values = get_histogram_features(feature_image, nbins=hist_bins,
                                           bins_range=hist_range)
    hist_features = np.concatenate((hist_values[0][0], hist_values[1][0],
                                    hist_values[2][0]))
    # Append the new feature vector to the features list
    hog_features = get_hog_features(feature_image, orient, pix_per_cell,
                                    cell_per_block)
    # assuming hls color space
    return (np.concatenate((spatial_features, hist_features, hog_features)))

def extract_features_for_data_set(data_set, input_data_obj, type='vehicle', 
                                  cspace='hls', spatial_size=(32, 32),
                                  hist_bins=32, hist_range=(0, 256),
                                  orient = 9, pix_per_cell=8, cell_per_block=2):
    features = []
    for each_data in data_set:
        if type == 'vehicle':
            each_image = input_data_obj.get_vehicle_image(each_data)
        elif type == 'non-vehicle':
            each_image = input_data_obj.get_non_vehicle_image(each_data)
        each_feature = extract_image_feature(each_image, cspace, spatial_size,
                                             hist_bins, hist_range, orient,
                                             pix_per_cell, cell_per_block)
        features.append(each_feature)
    return np.array(features)

# provide by udacity 
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# provide by udacity 
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def generate_heat_map(image_shape, car_windows):
    heat_map = np.zeros((image_shape[0], image_shape[1]))
    for each_car_window in car_windows:
        heat_map[each_car_window[0][1]:each_car_window[1][1], 
                 each_car_window[0][0]:each_car_window[1][0]] += 1
    return heat_map

def draw_labeled_bboxes(img, labels):
    image_copy = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image_copy, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return image_copy

class VehicleDetector(object):
    def __init__(self, trained_svm, feature_scaler,
                 cspace, spatial_size, hist_bins, hist_range, orient,
                 pix_per_cell, cell_per_block, x_start_stop,
                 y_start_stop, xy_scales, xy_overlap, heat_map_threshold):
        self.trained_svm = trained_svm
        self.feature_scaler = feature_scaler
        self.cspace = cspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.xy_scales = xy_scales
        self.xy_overlap = xy_overlap
        self.heat_map_threshold = heat_map_threshold
        self.heat_map_array = None
        self.heat_map_array_index = 0
        self.heat_map_list_length = 5
        

    def detect_in_images(self, image):
        sliding_windows = []
        for each_scale in self.xy_scales:
            sliding_windows.extend(slide_window(image, self.x_start_stop,
                                                self.y_start_stop, each_scale,
                                                self.xy_overlap))
        car_windows = []
        for each_window in sliding_windows:
            each_patch = image[each_window[0][1]:each_window[1][1], each_window[0][0]:each_window[1][0]]
            resized = cv2.resize(each_patch, (64,64))
            each_feature = extract_image_feature(resized, self.cspace, 
                                                 self.spatial_size,self.hist_bins, 
                                                 self.hist_range, self.orient,
                                                 self.pix_per_cell, self.cell_per_block)
            scaled_feature = self.feature_scaler.transform(each_feature.reshape(1, -1))
            if self.trained_svm.predict(scaled_feature) == 1:
                car_windows.append(each_window)

        heat_map = generate_heat_map(image.shape, car_windows)
        heat_map[heat_map<self.heat_map_threshold] = 0
        cars_found = label(heat_map)
        labeled_image = draw_labeled_bboxes(image, cars_found)
        return heat_map, labeled_image


    def detect_in_frames(self, image):
        sliding_windows = []
        for each_scale in self.xy_scales:
            sliding_windows.extend(slide_window(image, self.x_start_stop,
                                                self.y_start_stop, each_scale,
                                                self.xy_overlap))
        car_windows = []
        for each_window in sliding_windows:
            each_patch = image[each_window[0][1]:each_window[1][1], each_window[0][0]:each_window[1][0]]
            resized = cv2.resize(each_patch, (64,64))
            each_feature = extract_image_feature(resized, self.cspace, 
                                                 self.spatial_size,self.hist_bins, 
                                                 self.hist_range, self.orient,
                                                 self.pix_per_cell, self.cell_per_block)
            scaled_feature = self.feature_scaler.transform(each_feature.reshape(1, -1))
            if self.trained_svm.predict(scaled_feature) == 1:
                car_windows.append(each_window)

        heat_map = generate_heat_map(image.shape, car_windows)
        if self.heat_map_array == None:
             self.heat_map_array = np.zeros((image.shape[0],
                                             image.shape[1],
                                             self.heat_map_list_length))
        cur_index = self.heat_map_array_index % self.heat_map_list_length
        self.heat_map_array[:,:,cur_index] = heat_map
        sum_heat_map = np.sum(self.heat_map_array, axis=2)
        sum_heat_map[sum_heat_map<self.heat_map_threshold] = 0
        cars_found = label(sum_heat_map)
        labeled_image = draw_labeled_bboxes(image, cars_found)
        return sum_heat_map, labeled_image