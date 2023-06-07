```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

    2023-06-07 03:20:17.588496: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-06-07 03:20:17.588529: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/read_weights.py:28: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
      np.uint8, np.uint16, np.object, np.bool]
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[1], line 8
          5 import tensorflow as tf
          6 assert tf.__version__.startswith('2')
    ----> 8 from tflite_model_maker import model_spec
          9 from tflite_model_maker import image_classifier
         10 from tflite_model_maker.config import ExportFormat
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/__init__.py:44
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.
         16 
         17 You can install the package with
       (...)
         41 https://www.tensorflow.org/lite/guide/model_maker.
         42 """
    ---> 44 from tflite_model_maker import audio_classifier
         45 from tflite_model_maker import config
         46 from tflite_model_maker import image_classifier
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/audio_classifier/__init__.py:24
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """APIs to train an audio classification model.
         16 
         17 Tutorial:
       (...)
         21 https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/audio_classification_demo.py
         22 """
    ---> 24 from tensorflow_examples.lite.model_maker.core.data_util.audio_dataloader import DataLoader
         25 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import AudioClassifier
         26 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import create
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py:27
         25 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
         26 from tensorflow_examples.lite.model_maker.core.data_util import dataloader
    ---> 27 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         29 error_import_librosa = None
         30 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/__init__.py:20
         17 import inspect
         19 from tensorflow_examples.lite.model_maker.core.api import mm_export
    ---> 20 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         21 from tensorflow_examples.lite.model_maker.core.task.model_spec import image_spec
         22 from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py:30
         28 import tensorflow as tf
         29 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
    ---> 30 from tensorflow_examples.lite.model_maker.core.task import model_util
         31 import tensorflow_hub as hub
         33 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_util.py:28
         25 import tensorflow as tf
         27 from tensorflow_examples.lite.model_maker.core import compat
    ---> 28 from tensorflowjs.converters import converter as tfjs_converter
         29 from tflite_support import metadata as _metadata
         31 DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/__init__.py:21
         18 from __future__ import print_function
         20 # pylint: disable=unused-imports
    ---> 21 from tensorflowjs import converters
         22 from tensorflowjs import quantization
         23 from tensorflowjs import version
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/__init__.py:21
         18 from __future__ import print_function
         20 # pylint: disable=unused-imports,line-too-long
    ---> 21 from tensorflowjs.converters.converter import convert
         22 from tensorflowjs.converters.keras_h5_conversion import save_keras_model
         23 from tensorflowjs.converters.keras_tfjs_loader import deserialize_keras_model
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/converter.py:35
         33 from tensorflowjs import version
         34 from tensorflowjs.converters import common
    ---> 35 from tensorflowjs.converters import keras_h5_conversion as conversion
         36 from tensorflowjs.converters import keras_tfjs_loader
         37 from tensorflowjs.converters import tf_saved_model_conversion_v2
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/keras_h5_conversion.py:33
         30 import h5py
         31 import numpy as np
    ---> 33 from tensorflowjs import write_weights  # pylint: disable=import-error
         34 from tensorflowjs.converters import common
         37 def normalize_weight_name(weight_name):
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/write_weights.py:25
         22 import tensorflow as tf
         24 from tensorflowjs import quantization
    ---> 25 from tensorflowjs import read_weights
         27 _OUTPUT_DTYPES = [np.float16, np.float32, np.int32, np.complex64,
         28                   np.uint8, np.uint16, np.bool, np.object]
         29 _AUTO_DTYPE_CONVERSION = {
         30     np.dtype(np.float16): np.float32,
         31     np.dtype(np.float64): np.float32,
         32     np.dtype(np.int64): np.int32,
         33     np.dtype(np.complex128): np.complex64}
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/read_weights.py:28
         24 import numpy as np
         25 from tensorflowjs import quantization
         27 _INPUT_DTYPES = [np.float16, np.float32, np.int32, np.complex64,
    ---> 28                  np.uint8, np.uint16, np.object, np.bool]
         30 # Number of bytes used to encode the length of a string in a string tensor.
         31 STRING_LENGTH_NUM_BYTES = 4
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/numpy/__init__.py:305, in __getattr__(attr)
        300     warnings.warn(
        301         f"In the future `np.{attr}` will be defined as the "
        302         "corresponding NumPy scalar.", FutureWarning, stacklevel=2)
        304 if attr in __former_attrs__:
    --> 305     raise AttributeError(__former_attrs__[attr])
        307 # Importing Tester requires importing all of UnitTest which is not a
        308 # cheap import Since it is mainly used in test suits, we lazy import it
        309 # here to save on the order of 10 ms of import time for most users
        310 #
        311 # The previous way Tester was imported also had a side effect of adding
        312 # the full `numpy.testing` namespace
        313 if attr == 'testing':
    

    AttributeError: module 'numpy' has no attribute 'object'.
    `np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe. 
    The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
        https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations



```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

    2023-06-07 03:26:04.163751: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-06-07 03:26:04.163797: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[1], line 8
          5 import tensorflow as tf
          6 assert tf.__version__.startswith('2')
    ----> 8 from tflite_model_maker import model_spec
          9 from tflite_model_maker import image_classifier
         10 from tflite_model_maker.config import ExportFormat
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/__init__.py:44
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.
         16 
         17 You can install the package with
       (...)
         41 https://www.tensorflow.org/lite/guide/model_maker.
         42 """
    ---> 44 from tflite_model_maker import audio_classifier
         45 from tflite_model_maker import config
         46 from tflite_model_maker import image_classifier
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/audio_classifier/__init__.py:24
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """APIs to train an audio classification model.
         16 
         17 Tutorial:
       (...)
         21 https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/audio_classification_demo.py
         22 """
    ---> 24 from tensorflow_examples.lite.model_maker.core.data_util.audio_dataloader import DataLoader
         25 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import AudioClassifier
         26 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import create
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py:27
         25 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
         26 from tensorflow_examples.lite.model_maker.core.data_util import dataloader
    ---> 27 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         29 error_import_librosa = None
         30 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/__init__.py:20
         17 import inspect
         19 from tensorflow_examples.lite.model_maker.core.api import mm_export
    ---> 20 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         21 from tensorflow_examples.lite.model_maker.core.task.model_spec import image_spec
         22 from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py:30
         28 import tensorflow as tf
         29 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
    ---> 30 from tensorflow_examples.lite.model_maker.core.task import model_util
         31 import tensorflow_hub as hub
         33 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_util.py:29
         27 from tensorflow_examples.lite.model_maker.core import compat
         28 from tensorflowjs.converters import converter as tfjs_converter
    ---> 29 from tflite_support import metadata as _metadata
         31 DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0
         32 ESTIMITED_STEPS_PER_EPOCH = 1000
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_support/__init__.py:48
         46 from tensorflow_lite_support.metadata import metadata_schema_py_generated
         47 from tensorflow_lite_support.metadata import schema_py_generated
    ---> 48 from tensorflow_lite_support.metadata.python import metadata
         49 from tflite_support import metadata_writers
         51 if platform.system() != 'Windows':
         52   # Task Library is not supported on Windows yet.
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_lite_support/metadata/python/metadata.py:30
         28 from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
         29 from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb
    ---> 30 from tensorflow_lite_support.metadata.cc.python import _pywrap_metadata_version
         31 from tensorflow_lite_support.metadata.flatbuffers_lib import _pywrap_flatbuffers
         33 try:
         34   # If exists, optionally use TensorFlow to open and check files. Used to
         35   # support more than local file systems.
         36   # In pip requirements, we doesn't necessarily need tensorflow as a dep.
    

    ImportError: libusb-1.0.so.0: cannot open shared object file: No such file or directory



```python
sudo apt update && sudo apt install libusb-1.0-0
```


      Cell In[2], line 1
        sudo apt update && sudo apt install libusb-1.0-0
             ^
    SyntaxError: invalid syntax
    



```python
pip install numpy==1.23
```

    Collecting numpy==1.23
      Downloading numpy-1.23.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m55.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.24.3
        Uninstalling numpy-1.24.3:
          Successfully uninstalled numpy-1.24.3
    Successfully installed numpy-1.23.0
    Note: you may need to restart the kernel to use updated packages.
    


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/read_weights.py:28: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
      np.uint8, np.uint16, np.object, np.bool]
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[3], line 8
          5 import tensorflow as tf
          6 assert tf.__version__.startswith('2')
    ----> 8 from tflite_model_maker import model_spec
          9 from tflite_model_maker import image_classifier
         10 from tflite_model_maker.config import ExportFormat
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/__init__.py:44
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.
         16 
         17 You can install the package with
       (...)
         41 https://www.tensorflow.org/lite/guide/model_maker.
         42 """
    ---> 44 from tflite_model_maker import audio_classifier
         45 from tflite_model_maker import config
         46 from tflite_model_maker import image_classifier
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tflite_model_maker/audio_classifier/__init__.py:24
          1 # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
          2 #
          3 # Licensed under the Apache License, Version 2.0 (the "License");
       (...)
         13 # limitations under the License.
         14 # pylint: disable=g-bad-import-order,redefined-builtin
         15 """APIs to train an audio classification model.
         16 
         17 Tutorial:
       (...)
         21 https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/audio_classification_demo.py
         22 """
    ---> 24 from tensorflow_examples.lite.model_maker.core.data_util.audio_dataloader import DataLoader
         25 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import AudioClassifier
         26 from tensorflow_examples.lite.model_maker.core.task.audio_classifier import create
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/data_util/audio_dataloader.py:27
         25 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
         26 from tensorflow_examples.lite.model_maker.core.data_util import dataloader
    ---> 27 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         29 error_import_librosa = None
         30 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/__init__.py:20
         17 import inspect
         19 from tensorflow_examples.lite.model_maker.core.api import mm_export
    ---> 20 from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
         21 from tensorflow_examples.lite.model_maker.core.task.model_spec import image_spec
         22 from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_spec/audio_spec.py:30
         28 import tensorflow as tf
         29 from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
    ---> 30 from tensorflow_examples.lite.model_maker.core.task import model_util
         31 import tensorflow_hub as hub
         33 try:
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_examples/lite/model_maker/core/task/model_util.py:28
         25 import tensorflow as tf
         27 from tensorflow_examples.lite.model_maker.core import compat
    ---> 28 from tensorflowjs.converters import converter as tfjs_converter
         29 from tflite_support import metadata as _metadata
         31 DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/__init__.py:21
         18 from __future__ import print_function
         20 # pylint: disable=unused-imports
    ---> 21 from tensorflowjs import converters
         22 from tensorflowjs import quantization
         23 from tensorflowjs import version
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/__init__.py:21
         18 from __future__ import print_function
         20 # pylint: disable=unused-imports,line-too-long
    ---> 21 from tensorflowjs.converters.converter import convert
         22 from tensorflowjs.converters.keras_h5_conversion import save_keras_model
         23 from tensorflowjs.converters.keras_tfjs_loader import deserialize_keras_model
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/converter.py:35
         33 from tensorflowjs import version
         34 from tensorflowjs.converters import common
    ---> 35 from tensorflowjs.converters import keras_h5_conversion as conversion
         36 from tensorflowjs.converters import keras_tfjs_loader
         37 from tensorflowjs.converters import tf_saved_model_conversion_v2
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/converters/keras_h5_conversion.py:33
         30 import h5py
         31 import numpy as np
    ---> 33 from tensorflowjs import write_weights  # pylint: disable=import-error
         34 from tensorflowjs.converters import common
         37 def normalize_weight_name(weight_name):
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/write_weights.py:25
         22 import tensorflow as tf
         24 from tensorflowjs import quantization
    ---> 25 from tensorflowjs import read_weights
         27 _OUTPUT_DTYPES = [np.float16, np.float32, np.int32, np.complex64,
         28                   np.uint8, np.uint16, np.bool, np.object]
         29 _AUTO_DTYPE_CONVERSION = {
         30     np.dtype(np.float16): np.float32,
         31     np.dtype(np.float64): np.float32,
         32     np.dtype(np.int64): np.int32,
         33     np.dtype(np.complex128): np.complex64}
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflowjs/read_weights.py:28
         24 import numpy as np
         25 from tensorflowjs import quantization
         27 _INPUT_DTYPES = [np.float16, np.float32, np.int32, np.complex64,
    ---> 28                  np.uint8, np.uint16, np.object, np.bool]
         30 # Number of bytes used to encode the length of a string in a string tensor.
         31 STRING_LENGTH_NUM_BYTES = 4
    

    File /opt/conda/envs/tf/lib/python3.8/site-packages/numpy/__init__.py:305, in __getattr__(attr)
        300     warnings.warn(
        301         f"In the future `np.{attr}` will be defined as the "
        302         "corresponding NumPy scalar.", FutureWarning, stacklevel=2)
        304 if attr in __former_attrs__:
    --> 305     raise AttributeError(__former_attrs__[attr])
        307 # Importing Tester requires importing all of UnitTest which is not a
        308 # cheap import Since it is mainly used in test suits, we lazy import it
        309 # here to save on the order of 10 ms of import time for most users
        310 #
        311 # The previous way Tester was imported also had a side effect of adding
        312 # the full `numpy.testing` namespace
        313 if attr == 'testing':
    

    AttributeError: module 'numpy' has no attribute 'object'.
    `np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe. 
    The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
        https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations



```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

    2023-06-07 03:31:27.997528: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-06-07 03:31:27.997582: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.10.0 and strictly below 2.13.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    /opt/conda/envs/tf/lib/python3.8/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228818944/228813984 [==============================] - 9s 0us/step
    228827136/228813984 [==============================] - 9s 0us/step
    


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

```

    2023-06-07 03:32:22.379924: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/envs/tf/lib/python3.8/site-packages/cv2/../../lib64:
    2023-06-07 03:32:22.379973: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2023-06-07 03:32:22.379999: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-a1a9dc): /proc/driver/nvidia/version does not exist
    2023-06-07 03:32:22.392295: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    


```python
loss, accuracy = model.evaluate(test_data)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 loss, accuracy = model.evaluate(test_data)
    

    NameError: name 'model' is not defined



```python
model = image_classifier.create(train_data)

```

    INFO:tensorflow:Retraining the models...
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    2023-06-07 03:33:10.012193: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-06-07 03:33:10.360993: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-06-07 03:33:10.399226: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-06-07 03:33:10.434534: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 25690112 exceeds 10% of free system memory.
    2023-06-07 03:33:10.447244: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
    

    103/103 [==============================] - 61s 566ms/step - loss: 0.8642 - accuracy: 0.7737
    Epoch 2/5
    103/103 [==============================] - 56s 538ms/step - loss: 0.6493 - accuracy: 0.8944
    Epoch 3/5
    103/103 [==============================] - 55s 532ms/step - loss: 0.6166 - accuracy: 0.9202
    Epoch 4/5
    103/103 [==============================] - 54s 518ms/step - loss: 0.5940 - accuracy: 0.9360
    Epoch 5/5
    103/103 [==============================] - 54s 518ms/step - loss: 0.5853 - accuracy: 0.9345
    


```python
loss, accuracy = model.evaluate(test_data)

```

    12/12 [==============================] - 8s 469ms/step - loss: 0.6272 - accuracy: 0.9019
    


```python
model.export(export_dir='.')

```

    2023-06-07 03:45:50.935762: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    

    INFO:tensorflow:Assets written to: /tmp/tmp8op0oxei/assets
    

    INFO:tensorflow:Assets written to: /tmp/tmp8op0oxei/assets
    2023-06-07 03:45:55.526983: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2023-06-07 03:45:55.527141: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
    2023-06-07 03:45:55.576969: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
      function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 23.136ms.
      function_optimizer: function_optimizer did nothing. time = 0.01ms.
    
    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    2023-06-07 03:45:56.749216: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
    2023-06-07 03:45:56.749273: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in /tmp/tmpvwlfdplm/labels.txt
    

    INFO:tensorflow:Saving labels in /tmp/tmpvwlfdplm/labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
    
