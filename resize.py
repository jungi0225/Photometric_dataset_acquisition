import os
import rawpy
import imageio
from PIL import Image
import cv2

def convert_dng_to_png(input_folder, output_folder):
    # 출력 폴더가 존재하지 않으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 입력 폴더 내의 모든 파일을 순회합니다.
    for filename in os.listdir(input_folder):
        # DNG 파일인지 확인합니다.
        if filename.lower().endswith('.dng'):
            # 파일의 전체 경로를 구성합니다.
            input_path = os.path.join(input_folder, filename)
            
            try:
                # RawPy를 사용하여 DNG 파일 열기
                with rawpy.imread(input_path) as raw:
                    # RAW 데이터를 이미지로 변환
                    rgb_image = raw.postprocess()
                    
                    # 원래 파일 이름에서 확장자를 .png로 변경
                    new_filename = f"{os.path.splitext(filename)[0]}.png"
                    output_path = os.path.join(output_folder, new_filename)
                    
                    # PNG로 저장
                    imageio.imsave(output_path, rgb_image)
                    print(f"Image {filename} has been converted and saved as {new_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def resize_png_images(input_folder, output_folder, size=(504, 672)):
    # 출력 폴더가 존재하지 않으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 입력 폴더 내의 모든 파일을 순회합니다.
    for filename in os.listdir(input_folder):
        # PNG 파일인지 확인합니다.
        if filename.lower().endswith('.png'):
            # 파일의 전체 경로를 구성합니다.
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 이미지 열기
            image = cv2.imread(input_path)
            
            if image is not None:
                # 이미지 리사이즈
                resized_image = cv2.resize(image, size)
                
                # 리사이즈된 이미지 저장
                cv2.imwrite(output_path, resized_image)
                print(f"Image {filename} has been resized and saved to {output_path}")
            else:
                print(f"Failed to open image {filename}")


if __name__ == "__main__":

    dng_input_folder = "/Users/hongjungi/Library/CloudStorage/OneDrive-개인/POSTECH/4-1/graphics/dataset_acquisition/dataset/owl/dng"
    png_output_folder = "/Users/hongjungi/Library/CloudStorage/OneDrive-개인/POSTECH/4-1/graphics/dataset_acquisition/dataset/owl/png"
    resize_output_folder = "/Users/hongjungi/Library/CloudStorage/OneDrive-개인/POSTECH/4-1/graphics/dataset_acquisition/dataset/owl/resize"
    
    convert_dng_to_png(dng_input_folder,png_output_folder)
    #resize_png_images(png_output_folder, resize_output_folder)
 