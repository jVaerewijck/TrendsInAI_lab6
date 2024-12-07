import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2

nir = rasterio.open("./T32ULD_20241112T104301_B08_10m.jp2")
red = rasterio.open("./T32ULD_20241112T104301_B04_10m.jp2")
nir_data = nir.read(1).astype(float)
red_data = red.read(1).astype(float)

ndvi = (nir_data - red_data) / (nir_data + red_data)

with rasterio.open(
    "ndvi_output.tif",
    "w",
    driver="GTiff",
    height=ndvi.shape[0],
    width=ndvi.shape[1],
    count=1,
    dtype=ndvi.dtype,
    crs=nir.crs,
    transform=nir.transform,
) as dst:
    dst.write(ndvi, 1)

file_path = "ndvi_output.tif"
with rasterio.open(file_path) as src:
    ndvi = src.read(1)
ndvi = np.clip(ndvi, 0, 1)
# plt.figure(figsize=(10, 6))
# plt.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
# plt.colorbar(label="NDVI")
# plt.title("NDVI Image")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.show()

threshold = 0.2
segmented = (ndvi < threshold).astype(int)
# cmap = plt.cm.colors.ListedColormap(['green',(0.5, 1, 0.5, 1), 'white'])
# plt.imshow(segmented, cmap=cmap)
# plt.title('Segmentatie van zieke gebieden')
# plt.show()

segmented = np.uint8((ndvi < threshold) * 255)
contours, hierarchy = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
ndvi_copy = np.copy(ndvi)

total_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    total_area += area
pixel_size = 10
real_world_area = total_area * (pixel_size ** 2)
print(real_world_area)
cv2.drawContours(ndvi_copy, contours, -1, (255, 255, 0), 2)
# contourim = cv2.resize(ndvi_copy,(1080,1080))
# cv2.imshow('Contours',contourim)
plt.imshow(ndvi_copy, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('NDVI with Contours')
plt.axis('off')
plt.show()

green = rasterio.open("./T32ULD_20241112T104301_B03_10m.jp2")
green_data = green.read(1).astype(float)
# composite = np.stack([red_data, green_data, nir_data], axis=-1)
# composite_norm = composite / composite.max()
# plt.imshow(composite_norm)
# plt.title('Multispectrale composiet')
# plt.show()

blue = rasterio.open("./T32ULD_20241112T104301_B02_10m.jp2")
blue_data = blue.read(1).astype(float)

true_color = np.dstack((blue_data, green_data, red_data))
true_color = np.clip(true_color, 0, 1)
segmented
H, W, C = true_color.shape
X = true_color.reshape(-1, C)
y = segmented.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

predictions = model.predict(X).reshape(H, W)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(segmented, cmap='gray')
plt.title('NDVI Anomaly Labels (Ground Truth)')
plt.subplot(1, 2, 2)
plt.imshow(predictions, cmap='gray')
plt.title('Model Predicted Anomalies')
plt.show()

y_true = segmented.flatten()
y_pred = predictions.flatten()

print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))