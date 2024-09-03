import numpy as np
import cv2

# Load the pre-trained model
prototxt =  "C:/Users/raksh/OneDrive/Desktop/dl project/models/colorization_deploy_v2.prototxt"
points= "C:/Users/raksh/OneDrive/Desktop/dl project/models/colorization_release_v2.caffemodel"
model = "C:/Users/raksh/OneDrive/Desktop/dl project/models/pts_in_hull.npy"


net = cv2.dnn.readNetFromCaffe(prototxt, points)
pts = np.load(model)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input black and white image
image = cv2.imread("C:/Users/raksh/OneDrive/Desktop/dl project/images/13.jpg")

# Convert the image to LAB color space
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize the image to 224x224
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Colorize the image
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the colorized image to the original size
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Combine the colorized image with the original L channel
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert the colorized image back to BGR color space
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# Display the original and colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)