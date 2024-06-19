from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import numpy as np

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Next we compute the embeddings
# To encode an image, you can use the following code:
# from PIL import Image
# encoded_image = model.encode(Image.open(filepath))
base_image_names = list(glob.glob('./images_eye/*.PNG'))
base_encoded_images = np.array([model.encode(Image.open(x)) for x in base_image_names])

current_encoded_image = model.encode(Image.open('eye.png'))

# Now we run the clustering algorithm. This function compares images aganist 
# all other images and returns a list with the pairs that have the highest 
# cosine similarity score
ranks = model.similarity(current_encoded_image, base_encoded_images)
print(ranks)
NUM_SIMILAR_IMAGES = 10 

# =================
# DUPLICATES
# =================
# print('Finding duplicate images...')
# Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
# A duplicate image will have a score of 1.00
# It may be 0.9999 due to lossy image compression (.jpg)
# duplicates = [image for image in processed_images if image[0] >= 0.999]

# Output the top X duplicate images
# for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
#     print("\nScore: {:.3f}%".format(score * 100))
#     print(image_names[image_id1])
#     print(image_names[image_id2])

# =================
# NEAR DUPLICATES
# =================
# print('Finding near duplicate images...')
# Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
# A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
# threshold = 0.99
# near_duplicates = [image for image in processed_images if image[0] < threshold]
i = 0
for score in ranks[0]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(base_image_names[i])
    i += 1