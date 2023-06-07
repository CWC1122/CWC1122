```python
import os
import zipfile

local_zip = 'D:/mldownload/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/mldownload/')
zip_ref.close()

local_zip = 'D:/mldownload/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/mldownload/')
zip_ref.close()

```


```python
rock_dir = os.path.join('D:/mldownload/rps/rock')
paper_dir = os.path.join('D:/mldownload/rps/paper')
scissors_dir = os.path.join('D:/mldownload/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock01-000.png', 'rock01-001.png', 'rock01-002.png', 'rock01-003.png', 'rock01-004.png', 'rock01-005.png', 'rock01-006.png', 'rock01-007.png', 'rock01-008.png', 'rock01-009.png']
    ['paper01-000.png', 'paper01-001.png', 'paper01-002.png', 'paper01-003.png', 'paper01-004.png', 'paper01-005.png', 'paper01-006.png', 'paper01-007.png', 'paper01-008.png', 'paper01-009.png']
    ['scissors01-000.png', 'scissors01-001.png', 'scissors01-002.png', 'scissors01-003.png', 'scissors01-004.png', 'scissors01-005.png', 'scissors01-006.png', 'scissors01-007.png', 'scissors01-008.png', 'scissors01-009.png']
    


```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[3], line 1
    ----> 1 get_ipython().run_line_magic('matplotlib', 'inline')
          3 import matplotlib.pyplot as plt
          4 import matplotlib.image as mpimg
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\IPython\core\interactiveshell.py:2414, in InteractiveShell.run_line_magic(self, magic_name, line, _stack_depth)
       2412     kwargs['local_ns'] = self.get_local_scope(stack_depth)
       2413 with self.builtin_trap:
    -> 2414     result = fn(*args, **kwargs)
       2416 # The code below prevents the output from being displayed
       2417 # when using magics with decodator @output_can_be_silenced
       2418 # when the last Python token in the expression is a ';'.
       2419 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\IPython\core\magics\pylab.py:99, in PylabMagics.matplotlib(self, line)
         97     print("Available matplotlib backends: %s" % backends_list)
         98 else:
    ---> 99     gui, backend = self.shell.enable_matplotlib(args.gui.lower() if isinstance(args.gui, str) else args.gui)
        100     self._show_matplotlib_backend(args.gui, backend)
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\IPython\core\interactiveshell.py:3585, in InteractiveShell.enable_matplotlib(self, gui)
       3564 def enable_matplotlib(self, gui=None):
       3565     """Enable interactive matplotlib and inline figure support.
       3566 
       3567     This takes the following steps:
       (...)
       3583         display figures inline.
       3584     """
    -> 3585     from matplotlib_inline.backend_inline import configure_inline_support
       3587     from IPython.core import pylabtools as pt
       3588     gui, backend = pt.find_gui_and_backend(gui, self.pylab_gui_select)
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\matplotlib_inline\__init__.py:1
    ----> 1 from . import backend_inline, config  # noqa
          2 __version__ = "0.1.6"
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\matplotlib_inline\backend_inline.py:6
          1 """A matplotlib backend for publishing figures via display_data"""
          3 # Copyright (c) IPython Development Team.
          4 # Distributed under the terms of the BSD 3-Clause License.
    ----> 6 import matplotlib
          7 from matplotlib import colors
          8 from matplotlib.backends import backend_agg
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\matplotlib\__init__.py:107
        103 import warnings
        105 # cbook must import matplotlib only within function
        106 # definitions, so it is safe to import from it here.
    --> 107 from . import _api, cbook, docstring, rcsetup
        108 from matplotlib.cbook import MatplotlibDeprecationWarning, sanitize_sequence
        109 from matplotlib.cbook import mplDeprecation  # deprecated
    

    File D:\CodeProgram\anaconda3\envs\py39\lib\site-packages\matplotlib\cbook\__init__.py:28
         25 import warnings
         26 import weakref
    ---> 28 import numpy as np
         30 import matplotlib
         31 from matplotlib import _api, _c_internal_utils
    

    ModuleNotFoundError: No module named 'numpy'



```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

```


    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_0.png)
    



    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_1.png)
    



    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_2.png)
    



    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_3.png)
    



    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_4.png)
    



    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_3_5.png)
    



```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "D:/mldownload/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "D:/mldownload/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/25
    20/20 [==============================] - 56s 3s/step - loss: 1.2618 - accuracy: 0.3405 - val_loss: 1.0815 - val_accuracy: 0.4220
    Epoch 2/25
    20/20 [==============================] - 57s 3s/step - loss: 1.1051 - accuracy: 0.3952 - val_loss: 0.9877 - val_accuracy: 0.6452
    Epoch 3/25
    20/20 [==============================] - 56s 3s/step - loss: 1.0750 - accuracy: 0.4373 - val_loss: 1.0098 - val_accuracy: 0.5914
    Epoch 4/25
    20/20 [==============================] - 56s 3s/step - loss: 0.9367 - accuracy: 0.5710 - val_loss: 0.7837 - val_accuracy: 0.6156
    Epoch 5/25
    20/20 [==============================] - 57s 3s/step - loss: 0.7757 - accuracy: 0.6488 - val_loss: 0.3819 - val_accuracy: 0.7849
    Epoch 6/25
    20/20 [==============================] - 57s 3s/step - loss: 0.6535 - accuracy: 0.6885 - val_loss: 0.3575 - val_accuracy: 0.7527
    Epoch 7/25
    20/20 [==============================] - 60s 3s/step - loss: 0.5798 - accuracy: 0.7413 - val_loss: 0.2090 - val_accuracy: 0.9328
    Epoch 8/25
    20/20 [==============================] - 57s 3s/step - loss: 0.5267 - accuracy: 0.7944 - val_loss: 0.0954 - val_accuracy: 1.0000
    Epoch 9/25
    20/20 [==============================] - 62s 3s/step - loss: 0.4673 - accuracy: 0.8056 - val_loss: 0.4752 - val_accuracy: 0.7231
    Epoch 10/25
    20/20 [==============================] - 63s 3s/step - loss: 0.3117 - accuracy: 0.8762 - val_loss: 0.3949 - val_accuracy: 0.7957
    Epoch 11/25
    20/20 [==============================] - 59s 3s/step - loss: 0.3772 - accuracy: 0.8631 - val_loss: 0.1779 - val_accuracy: 0.9113
    Epoch 12/25
    20/20 [==============================] - 56s 3s/step - loss: 0.2415 - accuracy: 0.9008 - val_loss: 0.0361 - val_accuracy: 0.9973
    Epoch 13/25
    20/20 [==============================] - 56s 3s/step - loss: 0.2220 - accuracy: 0.9139 - val_loss: 0.0258 - val_accuracy: 1.0000
    Epoch 14/25
    20/20 [==============================] - 57s 3s/step - loss: 0.1870 - accuracy: 0.9369 - val_loss: 0.0163 - val_accuracy: 1.0000
    Epoch 15/25
    20/20 [==============================] - 61s 3s/step - loss: 0.1801 - accuracy: 0.9317 - val_loss: 0.0407 - val_accuracy: 0.9892
    Epoch 16/25
    20/20 [==============================] - 69s 3s/step - loss: 0.1709 - accuracy: 0.9369 - val_loss: 0.0628 - val_accuracy: 0.9785
    Epoch 17/25
    20/20 [==============================] - 61s 3s/step - loss: 0.2174 - accuracy: 0.9274 - val_loss: 0.0537 - val_accuracy: 0.9731
    Epoch 18/25
    20/20 [==============================] - 59s 3s/step - loss: 0.0979 - accuracy: 0.9714 - val_loss: 0.0119 - val_accuracy: 1.0000
    Epoch 19/25
    20/20 [==============================] - 58s 3s/step - loss: 0.1509 - accuracy: 0.9397 - val_loss: 0.0277 - val_accuracy: 0.9866
    Epoch 20/25
    20/20 [==============================] - 59s 3s/step - loss: 0.1361 - accuracy: 0.9476 - val_loss: 0.4111 - val_accuracy: 0.7876
    Epoch 21/25
    20/20 [==============================] - 59s 3s/step - loss: 0.0875 - accuracy: 0.9675 - val_loss: 0.0160 - val_accuracy: 1.0000
    Epoch 22/25
    20/20 [==============================] - 58s 3s/step - loss: 0.0968 - accuracy: 0.9694 - val_loss: 0.0260 - val_accuracy: 0.9946
    Epoch 23/25
    20/20 [==============================] - 59s 3s/step - loss: 0.1362 - accuracy: 0.9492 - val_loss: 0.0687 - val_accuracy: 0.9570
    Epoch 24/25
    20/20 [==============================] - 59s 3s/step - loss: 0.0495 - accuracy: 0.9833 - val_loss: 0.0097 - val_accuracy: 1.0000
    Epoch 25/25
    20/20 [==============================] - 56s 3s/step - loss: 0.1557 - accuracy: 0.9488 - val_loss: 0.0787 - val_accuracy: 0.9731
    


```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```


    
![png](https://github.com/CWC1122/CWC1122/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%94%C2%B7%E4%BA%8C/images/output_5_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python

```
