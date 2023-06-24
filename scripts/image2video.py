from PIL import Image
import os

if 0:
    folder_path = 'tmp/'  # Path to the folder containing the images
    image_extensions = '.png'  # List of valid image file extensions

    image_names = [
    ]

    for i in range(0, 12):
        image_names.append('test_virtual_map' + str(i) + image_extensions)

    images = []
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        w, h = image.size
        image = image.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
        images.append(image)

    output_file = "output.gif"  # Output file name

    images[0].save(output_file, format="GIF", append_images=images[1:], save_all=True, duration=200, loop=0)
