import numpy as np
import cv2
import librosa
import os

digit_separation = 75

# Path to audio files
audio_path = "./audio"


def mat(digit):
    lcd_patterns = {
        0: [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        1: [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        2: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        3: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        4: [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        5: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        6: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        7: [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        8: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        9: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]
    }

    return np.array(lcd_patterns[digit])


def create_digit_image(number):
    # Define dot matrix dimensions
    
    dot_spacing = 5
    dot_radius = 25
    dot_height = 5

    # Calculate image width
    image_width = 3 * (2*dot_radius + dot_spacing) + digit_separation

    # Create a black image with specified dimensions
    dot_matrix_image = np.zeros((300, image_width), dtype=np.uint8)

    # Define dot positions for each digit (0 to 9)
    digit_mat = mat(number)

    # Set white dots based on the digit
    for x in range(3):
        for y in range(5):
            if(digit_mat[y][x] == 0):
                continue

            x_coord = digit_separation + x * (2 * dot_radius + dot_spacing) + dot_radius
            y_coord = 10 + y * (2 * dot_radius + dot_spacing) + dot_radius
            cv2.circle(dot_matrix_image, (x_coord, y_coord), dot_radius, 255, -1)

    return dot_matrix_image


def num_image(num):
    # Splitting num into its component digits
    digits = [int(d) for d in str(num)]

    # Generating images for each digit
    digit_images = [create_digit_image(digit) for digit in digits]


    ending_space_img = np.zeros((300, digit_separation), dtype=np.uint8)
    digit_images.append(ending_space_img)

    # Combining images of component digits into a single image
    combined_image = np.hstack(digit_images)

    return combined_image

def evaluate_brick_quality(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Set parameters for spectrogram computation
    n_fft = 2048  # FFT points
    hop_length = 512  # Sliding amount for windowed FFT

    # Compute mel spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)

    # Perform power to decibel (dB) image transformation
    spec_db = librosa.power_to_db(spec)

    # Evaluate quality based on spectrogram
    quality = "metal" if np.mean(spec_db) > -30 else "cardboard"

    return quality


def get_rgb_value(x, y):
    # Define the mapping between x, y coordinates and RGB values
    r = (x + y) % 256
    g = (x - y) % 256
    b = (x * y) % 256

    return (r, g, b)


def colorized_image(image):
    # Get the dimensions of the image
    height, width = image.shape

    # Create a blank colored image with the same dimensions
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate through each pixel of the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel in the black and white image is white
            if image[y, x] == 255:
                # Get the RGB values for the pixel
                r, g, b = get_rgb_value(x, y)
                # Set the corresponding pixel in the colored image
                colored_image[y, x] = (b, r, g)

    return colored_image


sum = 0

for filename in os.listdir(audio_path):

    audio_file_path = os.path.join(audio_path, filename)

    quality = evaluate_brick_quality(audio_file_path)
    if quality == "metal":
        sum += 1
    elif quality == "cardboard":
        sum += 2

# Get the image for the sum
sum_image = num_image(sum)

final_image = colorized_image(sum_image)

# Display the sum image
cv2.imshow("Answer", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

