from PIL import Image
import os

if 1:
    folder_path = 'test/tmp1/'  # Path to the folder containing the images
    image_extensions = '.png'  # List of valid image file extensions

    image_names = [
    ]

    for i in range(0, 124):
        image_names.append('test_virtual_map' + str(i) + image_extensions)

    images = []
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        w, h = image.size
        image = image.resize((int(w / 4), int(h / 4)), Image.ANTIALIAS)
        images.append(image)

    output_file = "output.gif"  # Output file name

    images[0].save(output_file, format="GIF", append_images=images[1:], save_all=True, duration=200, loop=0)

if 0:
    folder_path = 'test/tmp/'  # Path to the folder containing the images
    image_extensions = '.png'  # List of valid image file extensions

    image_names = [
    ]

    for i in range(0, 148):
        image_names.append('test_virtual_map' + str(i) + image_extensions)

    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        # Open the image
        image = Image.open(image_path)

        # Get the dimensions of the image
        width, height = image.size

        # Calculate the coordinates to split the image
        split_x = 1300

        # Crop the right-hand side half of the image
        right_half = image.crop((split_x, 0, width, height))

        # Save the cropped image
        right_half.save("tmp/" + image_name)