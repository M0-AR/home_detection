# Download images by Bing-image-downloader or simple-image-download

# https://stackoverflow.com/questions/60370799/google-image-download-with-python-cannot-download-images
# Import the library
from bing_image_downloader import downloader

# Set the search query
query_string = "klassiske villa"

# Download the images
downloader.download(query_string, limit=10, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60)


# import simple_image_download.simple_image_download as simp
#
#
# my_downloader = simp.Downloader()
# my_downloader.search_urls('dog',limit=10, verbose=True)
#
# # Get List of Saved URLs in cache
# print(my_downloader.get_urls())

# TODO: look at here for further info:
#  https://stackoverflow.com/questions/60134947/why-couldnt-i-download-images-from-google-with-python


# Validating: The SSIM value ranges from -1 to 1, with 1 indicating a perfect match and values closer to 0 indicating
# less similarity. Can adjust the threshold value to determine how similar the two images need to be to be considered
# a match.
import cv2

# Load the images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute the structural similarity index (SSIM) between the two images
ssim = cv2.compare_ssim(gray_image1, gray_image2)

# Print the similarity index
print("The SSIM between the two images is:", ssim)
