from data_util import read_train_data, read_test_data
from data_util import mask_to_rle,resize,np
from model import get_unet
import pandas as pd
import os
import cv2
from PIL import Image
epochs = 2


# get train_data
train,train_mask = read_train_data()

# get test_data
test,test_size = read_test_data()
# path to save images

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\nTraining...")
u_net.fit(train,train_mask,batch_size=16,epochs=epochs)

print("Predicting")
# Predict on test data
test_mask = u_net.predict(test,verbose=1)

# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                       (test_size[i][0],test_size[i][1]), 
                                       mode='constant', preserve_range=True))


test_ids,rles = mask_to_rle(test_mask_upsampled)


cv2.imshow('test_mask-1', test_mask_upsampled[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(test_mask_upsampled))
path = r'C:\Users\akkia\OneDrive\Desktop\SegmentedImages'

for i in range (len(test_mask_upsampled)):
  cv2.imwrite(os.path.join(path, test_ids[i]+".png"),test_mask_upsampled[i])
  # Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

print("Data saved")
