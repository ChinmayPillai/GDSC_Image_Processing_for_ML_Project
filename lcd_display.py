import numpy as np
import cv2

num = 3


def mat(digit):
    lcd_patterns = {
        0: [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        1: [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        2: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        3: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        4: [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        5: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        6: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        7: [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]],
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


def hanoi(dig):
    if dig == 0:
        return 1
    else:
        return 2 * hanoi(dig - 1) + 1

num = hanoi(num)
digit_separation = 75

# Splitting num into its component digits
digits = [int(d) for d in str(num)]

# Generating images for each digit
digit_images = [create_digit_image(digit) for digit in digits]


ending_space_img = np.zeros((300, digit_separation), dtype=np.uint8)
digit_images.append(ending_space_img)

# Combining images of component digits into a single image
combined_image = np.hstack(digit_images)

# Displaying the combined image
cv2.imshow("Answer", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()







