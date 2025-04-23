from os.path import join
from tensorflow.python.keras.applications import ResNet50
from utile import read_and_prep_images
from utile import decode_predictions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# image dir
image_dir ="/home/pliu/Downloads/data_set/deep_learning/tensorflow/train"
# image_name list
img_list = ['ffe5f6d8e2bff356e9482a80a6e29aac.jpg','fff43b07992508bc822f33d8ffd902ae.jpg','ffe2ca6c940cddfee68fa3cc6c63213f.jpg']
# get the full path of image
img_paths= [join(image_dir,file_name) for file_name in img_list]

# import the model
my_model = ResNet50(weights='/home/pliu/Downloads/data_set/deep_learning/tensorflow/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

# call the python dependencies for display image and result
most_likely_labels = decode_predictions(preds, top=3, class_list_path='/home/pliu/Downloads/data_set/deep_learning/tensorflow/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()
    print(most_likely_labels[i])


