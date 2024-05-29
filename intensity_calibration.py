import numpy as np
import rawpy
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

def sphere_fitting(raw_file):
    with rawpy.imread(raw_file) as raw:
        raw_image = raw.raw_image_visible
        bayer_pattern = raw.raw_pattern
        height, width = raw_image.shape

        # R, G, B 채널 초기화
        red_channel = np.zeros_like(raw_image, dtype=np.float32)
        green_channel = np.zeros_like(raw_image, dtype=np.float32)
        blue_channel = np.zeros_like(raw_image, dtype=np.float32)

        # Bayer 필터 배열에 따라 각 픽셀을 R, G, B 채널로 분류
        for y in range(height):
            for x in range(width):
                if bayer_pattern[y % 2, x % 2] == 0:  # Red
                    red_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 1:  # Green on Red/Blue row
                    green_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 2:  # Blue
                    blue_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 3:  # Green on Green row
                    green_channel[y, x] = raw_image[y, x]
        
        data = []
        origin_x = 1520
        origin_y = 1933
        origin_z = 3109
        cos = 1
        # 모든 좌표에 대해 (x, y, z=0, intensity) 데이터를 구면 좌표계로 변환하여 생성
        for y in range(height):
            for x in range(width):
                red_intensity = red_channel[y, x]
                green_intensity = green_channel[y, x]
                blue_intensity = blue_channel[y, x]

                # 원점을 (1520, 1933, 3109)로 변환
                x_new = x - origin_x
                y_new = y - origin_y
                z_new = 0 + origin_z  # 얘만 + 하는 이유는 카메라 시점으로 바꿔야 하기 때문

                r = np.sqrt(x_new**2 + y_new**2 + z_new**2)
                theta = np.arccos(z_new/r)  
                cos = np.cos(theta)
                #r = r*1.68             # micro meter scale
                r = np.around(r, decimals=0)
                phi = np.arctan2(y_new, x_new)

                #data.append((r, theta, phi, red_intensity, green_intensity, blue_intensity))
                data.append((r, theta, phi, red_intensity/cos, green_intensity/cos, blue_intensity/cos))
        spherical_data_array = np.array(data)


        unique_r_values = np.unique(spherical_data_array[:, 0])
        avg_data = []
        for r_value in unique_r_values:
            mask = spherical_data_array[:, 0] == r_value
            subset = spherical_data_array[mask]

            # 각 채널에서 intensity 값이 700 이상인 값들만 선택하여 평균 계산
            valid_red = subset[subset[:, 3]*cos >= 700][:, 3]
            valid_green = subset[subset[:, 4]*cos>= 700][:, 4]
            valid_blue = subset[subset[:, 5]*cos >= 700][:, 5]

            avg_red = np.mean(valid_red) if len(valid_red) > 0 else 0
            avg_green = np.mean(valid_green) if len(valid_green) > 0 else 0
            avg_blue = np.mean(valid_blue) if len(valid_blue) > 0 else 0

            avg_data.append((r_value, avg_red, avg_green, avg_blue))

        avg_data_array = np.array(avg_data)

        #꼬리부분 제거
        valid_mask = (avg_data_array[:, 1] > 0) & (avg_data_array[:, 2] > 0) & (avg_data_array[:, 3] > 0)
        valid_avg_data_array = avg_data_array[valid_mask]

     
        def inverse_func(x, a, b):
            return a / (x**2) + b

        # 반비례 함수 피팅
        popt_red, _ = curve_fit(inverse_func, valid_avg_data_array[:, 0], valid_avg_data_array[:, 1])
        popt_green, _ = curve_fit(inverse_func, valid_avg_data_array[:, 0], valid_avg_data_array[:, 2])
        popt_blue, _ = curve_fit(inverse_func, valid_avg_data_array[:, 0], valid_avg_data_array[:, 3])

        print("Red Channel Fit Coefficients: a =", popt_red[0], ", b =", popt_red[1])
        print("Green Channel Fit Coefficients: a =", popt_green[0], ", b =", popt_green[1])
        print("Blue Channel Fit Coefficients: a =", popt_blue[0], ", b =", popt_blue[1])

        # 피팅 결과를 사용하여 그래프 그리기
        r_values = np.linspace(np.min(valid_avg_data_array[:, 0]), np.max(valid_avg_data_array[:, 0]), 500)

        fitted_red = inverse_func(r_values, *popt_red)
        fitted_green = inverse_func(r_values, *popt_green)
        fitted_blue = inverse_func(r_values, *popt_blue)

        plt.figure(figsize=(10, 6))
        plt.scatter(valid_avg_data_array[:, 0], valid_avg_data_array[:, 1], label='Red Channel Data', color='r', alpha=0.5)
        plt.scatter(valid_avg_data_array[:, 0], valid_avg_data_array[:, 2], label='Green Channel Data', color='g', alpha=0.5)
        plt.scatter(valid_avg_data_array[:, 0], valid_avg_data_array[:, 3], label='Blue Channel Data', color='b', alpha=0.5)
        plt.plot(r_values, fitted_red, label='Red Channel Fit', color='r')
        plt.plot(r_values, fitted_green, label='Green Channel Fit', color='g')
        plt.plot(r_values, fitted_blue, label='Blue Channel Fit', color='b')
        plt.xlabel('r Value')
        plt.ylabel('Average Intensity')
        plt.title('Average Intensity vs. r Value with Inverse Polynomial Fit')
        plt.legend()
        plt.grid(True)
        plt.show()


def fit_polynomial(x, y, z, degree):
    # 다항식 피팅을 위해 A 행렬 생성
    A = np.c_[np.ones(x.shape), x, y, x**2, x*y, y**2]
    B = z
    
    # 선형 회귀를 사용하여 다항식 계수 계산
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    return coeff

def calculate_fitted_surface(X, Y, coeff):
    return (coeff[0] + coeff[1]*X + coeff[2]*Y + coeff[3]*X**2 + coeff[4]*X*Y + coeff[5]*Y**2)

def plot_3d_intensity_with_fit(raw_file, degree=3, threshold=10):
    # RAW 파일을 열고 데이터를 읽음
    with rawpy.imread(raw_file) as raw:
        # gamma와 white balance를 적용하지 않고 postprocess
        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8)
        
        # x, y 좌표 생성
        height, width, _ = rgb_image.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # 각 채널의 intensity 값 추출
        red_channel = rgb_image[:, :, 0]
        green_channel = rgb_image[:, :, 1]
        blue_channel = rgb_image[:, :, 2]
        
        # 임계값 이상의 데이터만 선택
        def filter_and_fit(channel):
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = channel.flatten()
            mask = z_flat > threshold
            return fit_polynomial(x_flat[mask], y_flat[mask], z_flat[mask], degree)
        
        # 각 채널에 대해 다항식 피팅
        red_coeff = filter_and_fit(red_channel)
        green_coeff = filter_and_fit(green_channel)
        blue_coeff = filter_and_fit(blue_channel)
        
        # 피팅된 곡면 계산
        red_fit = calculate_fitted_surface(X, Y, red_coeff)
        green_fit = calculate_fitted_surface(X, Y, green_coeff)
        blue_fit = calculate_fitted_surface(X, Y, blue_coeff)
        
        red_fit = np.clip(red_fit, 0, None)
        green_fit = np.clip(green_fit, 0, None)
        blue_fit = np.clip(blue_fit, 0, None)

        # 3D 그래프 그리기
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 원래 Red 채널 및 피팅된 곡면
        #ax.plot_wireframe(X, Y, red_channel, color='r', alpha=0.5, label='Original Red Channel')
        ax.plot_wireframe(X, Y, red_fit, color='darkred', alpha=0.5, label='Fitted Red Surface')
        
        # 원래 Green 채널 및 피팅된 곡면
        #ax.plot_wireframe(X, Y, green_channel, color='g', alpha=0.5, label='Original Green Channel')
        #ax.plot_wireframe(X, Y, green_fit, color='darkgreen', alpha=0.5, label='Fitted Green Surface')
        
        # 원래 Blue 채널 및 피팅된 곡면
        #ax.plot_wireframe(X, Y, blue_channel, color='b', alpha=0.5, label='Original Blue Channel')
        #ax.plot_wireframe(X, Y, blue_fit, color='darkblue', alpha=0.5, label='Fitted Blue Surface')
        
        ax.set_zlim(0,None)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Intensity')
        ax.set_title('3D Intensity Plot of RGB Channels with Polynomial Fit')
        
        plt.legend()
        plt.show()

def make_vignetting_mask(raw_file):
    with rawpy.imread(raw_file) as raw:
    # gamma와 white balance를 적용하지 않고 postprocess
        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8)

def plot_3d_intensity(raw_file):
    # RAW 파일을 열고 데이터를 읽음
    with rawpy.imread(raw_file) as raw:
        # gamma와 white balance를 적용하지 않고 postprocess
        raw_image = raw.raw_image_visible
        bayer_pattern = raw.raw_pattern
        height, width = raw_image.shape

        # R, G, B 채널 초기화
        red_channel = np.zeros_like(raw_image, dtype=np.float32)
        green_channel = np.zeros_like(raw_image, dtype=np.float32)
        blue_channel = np.zeros_like(raw_image, dtype=np.float32)

        # Bayer 필터 배열에 따라 각 픽셀을 R, G, B 채널로 분류
        for y in range(height):
            for x in range(width):
                if bayer_pattern[y % 2, x % 2] == 0:  # Red
                    red_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 1:  # Green on Red/Blue row
                    green_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 2:  # Blue
                    blue_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 3:  # Green on Green row
                    green_channel[y, x] = raw_image[y, x]
        
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # 3D 그래프 그리기
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Red 채널
        #ax.plot_wireframe(X, Y, red_channel, color='r', label='Red Channel')
        # Green 채널
        #ax.plot_wireframe(X, Y, green_channel, color='g', label='Green Channel')
        # Blue 채널
        ax.scatter(X, Y, blue_channel, color='b', label='Blue Channel')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Intensity')
        ax.set_title('3D Intensity Plot of RGB Channels')
        
        plt.legend()
        plt.show()

def plot_intensity_difference(raw_file): 

    with rawpy.imread(raw_file) as raw: 

        raw_image = raw.raw_image_visible
        bayer_pattern = raw.raw_pattern
        height, width = raw_image.shape

        # R, G, B 채널 초기화
        red_channel = np.zeros_like(raw_image, dtype=np.float32)
        green_channel = np.zeros_like(raw_image, dtype=np.float32)
        blue_channel = np.zeros_like(raw_image, dtype=np.float32)

        # Bayer 필터 배열에 따라 각 픽셀을 R, G, B 채널로 분류
        for y in range(height):
            for x in range(width):
                if bayer_pattern[y % 2, x % 2] == 0:  # Red
                    red_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 1:  # Green on Red/Blue row
                    green_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 2:  # Blue
                    blue_channel[y, x] = raw_image[y, x]
                elif bayer_pattern[y % 2, x % 2] == 3:  # Green on Green row
                    green_channel[y, x] = raw_image[y, x]  


        x_center = 1521 

        y_coords = np.arange(height)
   
        plt.figure(figsize = (10,6))
        
        #plt.plot(y_coords, red_channel[:,x_center], color = 'red', label = 'Red Channel')
        #plt.plot(y_coords, green_channel[:,x_center], color= 'green', label = 'Green Channel')
        plt.plot(y_coords, blue_channel[:,x_center], color = 'blue', label= 'Blue Channel')
 

        plt.xlabel('Y Coordinate')
        plt.ylabel('Intensity')
        plt.title('Intensity Difference along Y-axis at X = 1521') 
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_intensity_difference_flat(raw_file):
    d = 3109
    y_center = 1933

    with rawpy.imread(raw_file) as raw:
        # gamma와 white balance를 적용하지 않고 postprocess
        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8)

        # 고정된 x축 좌표
        x_center = 1520
        
        # 각 채널의 intensity 값 추출
        red_channel = rgb_image[:, x_center, 0]
        green_channel = rgb_image[:, x_center, 1]
        blue_channel = rgb_image[:, x_center, 2]

        # y좌표 생성
        y_coords = np.arange(rgb_image.shape[0])

        # 수정된 intensity 값 계산
        distance_factors = np.sqrt( d**2 + np.abs(y_coords - y_center)**2 )
        print(distance_factors)  

        red_intensity_modified = red_channel * 3109 / distance_factors
        green_intensity_modified = green_channel * 3109 / distance_factors
        blue_intensity_modified = blue_channel * 3109/ distance_factors

        
        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, red_intensity_modified, color='red', label='Red Channel')
        plt.plot(y_coords, green_intensity_modified, color='green', label='Green Channel')
        plt.plot(y_coords, blue_intensity_modified, color='blue', label='Blue Channel')

        plt.xlabel('Y Coordinate')
        plt.ylabel('Modified Intensity')
        plt.title('Intensity Difference along Y-axis at X = 1520')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_intensity_difference_flat_rate(raw_file):
    d = 3109
    y_center = 1933

    with rawpy.imread(raw_file) as raw:
        # gamma와 white balance를 적용하지 않고 postprocess
        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8)

        # 고정된 x축 좌표
        x_center = 1520
        
        # 각 채널의 intensity 값 추출
        red_channel = rgb_image[:, x_center, 0]
        green_channel = rgb_image[:, x_center, 1]
        blue_channel = rgb_image[:, x_center, 2]

        # y좌표 생성
        y_coords = np.arange(rgb_image.shape[0])

        # 수정된 intensity 값 계산
        distance_factors = np.sqrt( d**2 + np.abs(y_coords - y_center)**2 )
        print(distance_factors)  

        red_intensity_modified = red_channel* 3109 / distance_factors
        green_intensity_modified = green_channel*3109 / distance_factors
        blue_intensity_modified = blue_channel*3109 / distance_factors
        #print(f"{red_intensity_modified[1933]}")
        #print(f"{green_intensity_modified[1933]}")
        #print(f"{blue_intensity_modified[1933]}")

        #rate_red_intensity_modified = red_intensity_modified/ red_intensity_modified[1933]
        #rate_green_intensity_modified = green_intensity_modified/green_intensity_modified[1933]
        #rate_blue_intensity_modified  = blue_intensity_modified / blue_intensity_modified[1933]
        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, red_intensity_modified, color='red', label='Red Channel')
        #plt.plot(y_coords, green_intensity_modified, color='green', label='Green Channel')
        #plt.plot(y_coords, blue_intensity_modified, color='blue', label='Blue Channel')

        plt.xlabel('Y Coordinate')
        plt.ylabel('Modified Intensity')
        plt.title('Intensity Difference along Y-axis at X = 1520')
        plt.legend()
        plt.grid(True)
        plt.show()


def find_rectangles_in_image(raw_file): 

    with rawpy.imread(raw_file) as raw: 

        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8)

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        _ ,binary_image = cv2.threshold(gray_image , 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for contour in contours:

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,epsilon,True)

            if len(approx) == 4:
                rectangles.append(approx)

        
        return rectangles

def find_max_intensity_pixel(raw_file):
    # RAW 파일을 열고 데이터를 읽음
    with rawpy.imread(raw_file) as raw:
        # gamma와 white balance를 적용하지 않고 postprocess
        
        rgb_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, output_bps=8 )
        print(f"shape of rgb_image{rgb_image.shape}")

        # 각 채널의 intensity 합산 (grayscale)
        intensity = np.sum(rgb_image, axis=2)
        
        # 최대 intensity 값을 가진 픽셀 찾기
        max_intensity_index = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
        max_intensity_value = intensity[max_intensity_index]
        return max_intensity_index, max_intensity_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the pixel with the highest intensity in a RAW image.')
    parser.add_argument('raw_file', type=str, help='Path to the RAW image file')
    args = parser.parse_args()
    max_intensity_pixel, max_intensity_value = find_max_intensity_pixel(args.raw_file)

    rectangles = find_rectangles_in_image(args.raw_file)

    
    print(f"Pixel with highest intensity: {max_intensity_pixel}")
    print(f"Maximum intensity value: {max_intensity_value}")

    #plot_3d_intensity(args.raw_file)
    #plot_3d_intensity_with_fit(args.raw_file,3,10)
    #plot_intensity_difference_flat(args.raw_file)
    #plot_intensity_difference(args.raw_file)
    #plot_intensity_difference_flat_rate(args.raw_file)

    sphere_fitting(args.raw_file)