

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Plot settings
# plt.rc('figure', autolayout=True)
# plt.rc('image', cmap='magma')

# # Define edge-detection kernel
# kernel = tf.constant([
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1]
# ], dtype=tf.float32)

# # Load image
# image = tf.io.read_file('iphone.jpg')
# image = tf.io.decode_image(image, channels=3)      # automatically detect format
# image = tf.image.rgb_to_grayscale(image)          # convert to 1 channel
# image = tf.image.resize(image, [300, 300])

# # Plot original image
# plt.figure(figsize=(5, 5))
# plt.imshow(tf.squeeze(image), cmap='gray')
# plt.axis('off')
# plt.title('Original Grayscale Image')
# plt.show()

# # Convert image to float32 and add batch dimension
# image = tf.image.convert_image_dtype(image, tf.float32)
# image = tf.expand_dims(image, axis=0)             # shape: (1, 300, 300, 1)

# # Reshape kernel to [filter_height, filter_width, in_channels, out_channels]
# kernel = tf.reshape(kernel, [3, 3, 1, 1])

# # Convolution
# image_conv = tf.nn.conv2d(
#     input=image,
#     filters=kernel,
#     strides=1,
#     padding='SAME'
# )

# # Activation (ReLU)
# image_relu = tf.nn.relu(image_conv)

# # Pooling (Max Pool 2x2)
# image_pool = tf.nn.pool(
#     input=image_relu,
#     window_shape=(2, 2),
#     pooling_type='MAX',
#     strides=(2, 2),
#     padding='SAME'
# )

# # Plot results
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(tf.squeeze(image_conv), cmap='gray')
# plt.axis('off')
# plt.title('Convolution')

# plt.subplot(1, 3, 2)
# plt.imshow(tf.squeeze(image_relu), cmap='gray')
# plt.axis('off')
# plt.title('Activation (ReLU)')

# plt.subplot(1, 3, 3)
# plt.imshow(tf.squeeze(image_pool), cmap='gray')
# plt.axis('off')
# plt.title('Pooling (2x2 Max)')

# plt.show()



# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Plot settings
# plt.rc('figure', autolayout=True)
# plt.rc('image', cmap='magma')

# # Define kernel (edge detection)
# kernel = tf.constant([
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1],
# ], dtype=tf.float32)

# # Load image
# image = tf.io.read_file('iphone.jpg')
# image = tf.io.decode_jpeg(image, channels=1)
# image = tf.image.resize(image, size=[300, 300])

# # Plot original image
# img = tf.squeeze(image).numpy()
# plt.figure(figsize=(5, 5))
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.title('Original Gray Scale Image')
# plt.show()

# # Reformat image
# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
# image = tf.expand_dims(image, axis=0)

# # Reformat kernel
# kernel = tf.reshape(kernel, [3, 3, 1, 1])

# # Convolution
# image_filter = tf.nn.conv2d(
#     input=image,
#     filters=kernel,
#     strides=[1, 1, 1, 1],   # corrected
#     padding='SAME'
# )

# # Activation (ReLU)
# image_detect = tf.nn.relu(image_filter)

# # Pooling
# image_condense = tf.nn.pool(
#     input=image_detect,
#     window_shape=(2, 2),
#     pooling_type='MAX',

    
#     strides=(2, 2),
#     padding='SAME'
# )

# # Plot results
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(tf.squeeze(image_filter).numpy(), cmap='gray')
# plt.axis('off')
# plt.title('Convolution')

# plt.subplot(1, 3, 2)
# plt.imshow(tf.squeeze(image_detect).numpy(), cmap='gray')
# plt.axis('off')
# plt.title('Activation (ReLU)')

# plt.subplot(1, 3, 3)
# plt.imshow(tf.squeeze(image_condense).numpy(), cmap='gray')
# plt.axis('off')
# plt.title('Pooling')

# plt.show()






import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot settings
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# Define edge detection kernel
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=tf.float32)

# -----------------------------
# Load image (supports JPEG, PNG, WebP)
# -----------------------------
image = tf.io.read_file('iphone.jpg')
image = tf.io.decode_image(image, channels=3, expand_animations=False)  # works for JPEG, PNG, WebP
image = tf.image.rgb_to_grayscale(image)  # convert to 1 channel
image = tf.image.resize(image, [300, 300])

# Plot original image
plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Original Grayscale Image')
plt.show()

# -----------------------------
# Preprocess image for conv2d
# -----------------------------
image = tf.image.convert_image_dtype(image, tf.float32)  # float32 [0,1]
image = tf.expand_dims(image, axis=0)                    # add batch dimension: shape (1, 300, 300, 1)

# -----------------------------
# Prepare kernel
# -----------------------------
kernel = tf.reshape(kernel, [3, 3, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]

# -----------------------------
# Convolution
# -----------------------------
image_conv = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)

# -----------------------------
# Activation (ReLU)
# -----------------------------
image_relu = tf.nn.relu(image_conv)

# -----------------------------
# Pooling (2x2 Max Pool)
# -----------------------------
image_pool = tf.nn.pool(
    input=image_relu,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME'
)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_conv), cmap='gray')
plt.axis('off')
plt.title('Convolution')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_relu), cmap='gray')
plt.axis('off')
plt.title('Activation (ReLU)')

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_pool), cmap='gray')
plt.axis('off')
plt.title('Pooling (2x2 Max)')

plt.show()
