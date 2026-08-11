API and Backend Reference
=========================

   Detailed documentation for all CLI entrypoints, Python module APIs, data structures (HDF5), model classes, evaluation scripts, and progress streaming.

--------------

Table of Contents
-----------------

1. `CLI Entrypoints <#cli-entrypoints>`__
2. `Python Module APIs <#python-module-apis>`__
3. `Data Structures <#data-structures>`__
4. `Model Class Reference <#model-class-reference>`__
5. `Evaluation Script Reference <#evaluation-script-reference>`__
6. `Progress Streaming <#progress-streaming>`__
7. `Environment Configuration <#environment-configuration>`__

--------------

CLI Entrypoints
---------------

   **Source**: ``aceverify/pyproject.toml:26``

The ``pip install -e .`` command installs three console scripts:

+--------------------------+-------------------------------+-------------------------------------------------+
| Command                  | Module:Function               | Description                                     |
+==========================+===============================+=================================================+
| ``aceverify-train``      | ``aceverify.train:main``      | Train the ACEVerifyModel on HDF5 datasets       |
+--------------------------+-------------------------------+-------------------------------------------------+
| ``aceverify-preprocess`` | ``aceverify.preprocess:main`` | Convert DFDC-style zip archives into HDF5 files |
+--------------------------+-------------------------------+-------------------------------------------------+
| ``aceverify-evaluate``   | ``evaluation.evaluate:main``  | Evaluate a trained checkpoint on test data      |
+--------------------------+-------------------------------+-------------------------------------------------+

--------------

``aceverify-train``
~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/train.py:216``

Trains the ``ACEVerifyModel`` on preprocessed HDF5 data.

.. code:: bash

   aceverify-train \
     --train_path data/train_data-003.h5 \
     --test_path data/test_data.h5 \
     --checkpoint-path results/aceverify_final.pth \
     --epochs 10 \
     --batch-size 8 \
     --log-level INFO

Arguments
^^^^^^^^^

+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| Argument              | Type        | Required    | Default                         | Description                                                                  |
+=======================+=============+=============+=================================+==============================================================================+
| ``--train_path``      | ``str``     | Yes         | —                               | Path to the training HDF5 file                                               |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| ``--test_path``       | ``str``     | Yes         | —                               | Path to the test HDF5 file                                                   |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| ``--checkpoint-path`` | ``str``     | No          | ``results/aceverify_final.pth`` | Where to save the final checkpoint                                           |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| ``--epochs``          | ``int``     | No          | ``10``                          | Number of training epochs                                                    |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| ``--batch-size``      | ``int``     | No          | ``8``                           | Training and validation batch size                                           |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+
| ``--log-level``       | ``str``     | No          | ``INFO``                        | Logging verbosity: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL`` |
+-----------------------+-------------+-------------+---------------------------------+------------------------------------------------------------------------------+

Behavior
^^^^^^^^

1. Creates the checkpoint directory if it does not exist.
2. Instantiates ``ACEVerifyModel()`` and moves it to the device (``cuda`` if available, else ``cpu``).
3. Uses ``BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))`` as the loss function.
4. Uses ``AdamW(lr=5e-5, weight_decay=1e-4)`` as the optimizer.
5. Uses ``StepLR(step_size=2, gamma=0.5)`` as the learning rate scheduler.
6. Attempts to resume from an existing checkpoint (supports both ``state_dict`` and ``{'model_state_dict': ...}`` formats).
7. For each epoch:

   -  Rebuilds the training dataset (1000 samples, balanced real/fake, shuffled).
   -  Runs forward pass, computes loss, backpropagates, and updates weights.
   -  Runs validation on a held-out test dataset (200 samples, balanced).
   -  Logs training and validation accuracy.

8. After all epochs, generates a ``classification_report`` and saves the checkpoint and metrics CSV.

Outputs
^^^^^^^

+-----------------------+-----------------------------------------------+--------------------------------------------------------------------------+
| File                  | Location                                      | Description                                                              |
+=======================+===============================================+==========================================================================+
| Model checkpoint      | ``{checkpoint-path}``                         | ``model.state_dict()`` saved via ``torch.save()``                        |
+-----------------------+-----------------------------------------------+--------------------------------------------------------------------------+
| Metrics CSV           | ``{checkpoint-path without ext}_metrics.csv`` | Per-epoch columns: ``epochs``, ``train_accuracies``, ``test_accuracies`` |
+-----------------------+-----------------------------------------------+--------------------------------------------------------------------------+

--------------

``aceverify-preprocess``
~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/preprocess.py:217``

Converts DFDC-style zip archives into HDF5 datasets.

.. code:: bash

   aceverify-preprocess \
     dfdc_train_part_00.zip \
     dfdc_train_part_0 \
     --output data/processed_data.h5 \
     --temp-dir temp \
     --ffmpeg-bin /usr/bin/ffmpeg \
     --log-level INFO

.. _arguments-1:

Arguments
^^^^^^^^^

+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| Argument              | Type                 | Required    | Default                                             | Description                                              |
+=======================+======================+=============+=====================================================+==========================================================+
| ``zip_file``          | ``str`` (positional) | Yes         | —                                                   | Path to the DFDC-style zip archive                       |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| ``subfolder``         | ``str`` (positional) | Yes         | —                                                   | Subfolder name within the zip for temp extraction prefix |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| ``--output`` / ``-o`` | ``str``              | No          | ``processed_data.h5``                               | Output HDF5 file path                                    |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| ``--temp-dir``        | ``str``              | No          | ``<repo>/temp``                                     | Directory for temporary extracted/processed files        |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| ``--ffmpeg-bin``      | ``str``              | No          | ``None`` (resolves from ``FFMPEG_BIN`` or ``PATH``) | Path to ffmpeg executable                                |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+
| ``--log-level``       | ``str``              | No          | ``INFO``                                            | Logging verbosity                                        |
+-----------------------+----------------------+-------------+-----------------------------------------------------+----------------------------------------------------------+

.. _behavior-1:

Behavior
^^^^^^^^

1. Creates the output HDF5 file in write mode.
2. Reads the zip archive and finds the JSON label file (first ``.json`` entry).
3. Parses label mappings: ``{video_filename: {"label": "FAKE"|"REAL"}}``.
4. For each ``.mp4`` file in the zip:

   -  Extracts the video to the temp directory.
   -  Uses ffmpeg to extract 16 frames at 5s offset (``-ss 00:00:05 -frames:v 16 -q:v 2``).
   -  Uses ffmpeg to extract 0.5s audio at 5s offset (``-ss 00:00:05 -t 0.5 -acodec pcm_s16le``).
   -  Runs MTCNN face detection on frames 1-16 until a face is found.
   -  Expands the face bounding box by +/-80px (x) and +/-50px (y).
   -  Crops and resizes all 16 frames to the face region.
   -  Saves the 16-frame video and audio to the HDF5 file under a group named after the video.
   -  Cleans up temporary files.

5. If no face is detected in any frame, the video is skipped with a warning.

HDF5 Output Structure
^^^^^^^^^^^^^^^^^^^^^

::

   /
   ├── video_name_1/
   │   ├── video          # Dataset: [16, 224, 224, 3] uint8, gzip-compressed
   │   ├── audio          # Dataset: [N] float32, gzip-compressed
   │   └── (attrs)
   │       └── label      # 1 (FAKE) or 0 (REAL)
   ├── video_name_2/
   │   ├── ...

--------------

``aceverify-evaluate``
~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``evaluation/evaluate.py:55``

Evaluates a trained checkpoint on a test HDF5 file.

.. code:: bash

   aceverify-evaluate \
     --h5 data/test_data.h5 \
     --checkpoint results/aceverify_final.pth \
     --batch-size 8 \
     --log-level INFO

.. _arguments-2:

Arguments
^^^^^^^^^

+------------------+-------------+-------------+-------------+------------------------------+
| Argument         | Type        | Required    | Default     | Description                  |
+==================+=============+=============+=============+==============================+
| ``--h5``         | ``str``     | Yes         | —           | Path to the test HDF5 file   |
+------------------+-------------+-------------+-------------+------------------------------+
| ``--checkpoint`` | ``str``     | Yes         | —           | Path to the model checkpoint |
+------------------+-------------+-------------+-------------+------------------------------+
| ``--batch-size`` | ``int``     | No          | ``8``       | Inference batch size         |
+------------------+-------------+-------------+-------------+------------------------------+
| ``--log-level``  | ``str``     | No          | ``INFO``    | Logging verbosity            |
+------------------+-------------+-------------+-------------+------------------------------+

.. _behavior-2:

Behavior
^^^^^^^^

1. Loads the ``ACEDataset`` from the HDF5 file in non-training mode.
2. Creates a ``DataLoader`` with ``batch_size``, ``num_workers=2``, ``pin_memory=True``.
3. Instantiates ``ACEVerifyModel()`` and loads the checkpoint state dict.
4. Runs inference with ``torch.no_grad()`` and applies a sigmoid threshold of ``0.5``.
5. Saves predictions to CSV at ``{checkpoint_path without ext}_eval.csv``.

Output
^^^^^^

+-----------------------+--------------------------------------------+----------------------------------------------------------------+
| File                  | Location                                   | Description                                                    |
+=======================+============================================+================================================================+
| Predictions CSV       | ``{checkpoint_path without ext}_eval.csv`` | Columns: ``label`` (ground truth), ``pred`` (model prediction) |
+-----------------------+--------------------------------------------+----------------------------------------------------------------+

--------------

Python Module APIs
------------------

``aceverify`` Package
~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/__init__.py``

.. code:: python

   from .dataset import ACEDataset
   from .model import ACEVerifyModel
   from .train import train_model, load_data
   from .visualize_data import test_visualization, numRealAndFake
   from .preprocess import preprocess_dataset

   __all__ = [
       'ACEDataset',
       'ACEVerifyModel',
       'train_model',
       'load_data',
       'test_visualization',
       'numRealAndFake',
       'preprocess_dataset',
   ]

--------------

``aceverify.model.ACEVerifyModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/model.py:43``

Constructor
^^^^^^^^^^^

.. code:: python

   ACEVerifyModel()

Initializes the multimodal architecture with the following components:

+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Component        | Class/Type                                                                                         | Dimensions                                       | Trainable          |
+==================+====================================================================================================+==================================================+====================+
| Video backbone   | ``timm vit_base_patch16_224`` (ViT-B/16)                                                           | Input: ``[B, C, H, W]``, Output: 768             | Last 4 blocks only |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Temporal layer   | ``nn.GRU(768, 512, bidirectional=True)``                                                           | Input: ``[B, T, 768]``, Output: ``[B, T, 1024]`` | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Temporal pool    | ``TemporalAttentionPooling(1024)``                                                                 | Input: ``[B, T, 1024]``, Output: ``[B, 1024]``   | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Video projection | ``LayerNorm + Linear(1024,256) + GELU + Dropout(0.2)``                                             | Output: 256                                      | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Audio encoder    | ``SpectrogramEncoder()`` (EfficientNet-B0)                                                         | Input: ``[B, 1, H, W]``, Output: 256             | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Fusion gate      | ``Linear(512,256) + GELU + Linear(256,256) + Sigmoid``                                             | Output: 256                                      | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+
| Classifier       | ``Linear(1024,512) + GELU + Dropout(0.4) + Linear(512,128) + GELU + Dropout(0.2) + Linear(128,1)`` | Output: 1                                        | Yes                |
+------------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+--------------------+

Forward Method
^^^^^^^^^^^^^^

.. code:: python

   def forward(self, video, audio_spec=None):

+-----------------+--------------------------+-------------------------------------------------------------+-----------------------------------------------------------------+
| Parameter       | Type                     | Shape                                                       | Description                                                     |
+=================+==========================+=============================================================+=================================================================+
| ``video``       | ``torch.Tensor``         | ``[B, C, T, H, W]``                                         | Video frames tensor (channels, temporal, height, width)         |
+-----------------+--------------------------+-------------------------------------------------------------+-----------------------------------------------------------------+
| ``audio_spec``  | ``torch.Tensor \| None`` | ``[B, C, T, H, W]`` or ``[B, T, H, W]`` or ``[B, 1, H, W]`` | Mel-spectrogram tensor; if ``None``, a zero spectrogram is used |
+-----------------+--------------------------+-------------------------------------------------------------+-----------------------------------------------------------------+

**Returns**: ``torch.Tensor`` of shape ``[B, 1]`` containing the raw classification logit.

Fusion Math
^^^^^^^^^^^

.. code:: python

   gate = self.fusion_gate(fusion_input)                          # [B, 256], sigmoid-weighted
   fused_features = torch.cat([
       video_combined * gate,                                     # Gated video
       audio_features * (1.0 - gate),                             # Gated audio
       video_combined - audio_features,                           # Video-audio difference
       video_combined * audio_features,                           # Video-audio product
   ], dim=-1)                                                     # [B, 1024]

--------------

``aceverify.dataset.ACEDataset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/dataset.py:9``

.. code:: python

   ACEDataset(h5_path, indices=None, is_training=False)

+-----------------+-----------------------+-----------------+------------------------------------------------------+
| Parameter       | Type                  | Default         | Description                                          |
+=================+=======================+=================+======================================================+
| ``h5_path``     | ``str``               | *required*      | Path to the HDF5 file                                |
+-----------------+-----------------------+-----------------+------------------------------------------------------+
| ``indices``     | ``list[int] \| None`` | ``None``        | Sample indices to include; ``None`` uses all samples |
+-----------------+-----------------------+-----------------+------------------------------------------------------+
| ``is_training`` | ``bool``              | ``False``       | If ``True``, applies data augmentation               |
+-----------------+-----------------------+-----------------+------------------------------------------------------+

``__getitem__(idx)``
^^^^^^^^^^^^^^^^^^^^

Returns a tuple ``(video, spec, label)``:

+-----------------+------------------+------------------------------------------+-----------------------------------------+
| Element         | Type             | Shape                                    | Description                             |
+=================+==================+==========================================+=========================================+
| ``video``       | ``torch.Tensor`` | ``[C, T, H, W]`` = ``[3, 16, 224, 224]`` | Normalized video frames in ``[0, 1]``   |
+-----------------+------------------+------------------------------------------+-----------------------------------------+
| ``spec``        | ``torch.Tensor`` | ``[1, 1, 224, 224]``                     | Mel-spectrogram interpolated to 224x224 |
+-----------------+------------------+------------------------------------------+-----------------------------------------+
| ``label``       | ``torch.Tensor`` | scalar ``long``                          | ``0`` (Real) or ``1`` (Fake)            |
+-----------------+------------------+------------------------------------------+-----------------------------------------+

--------------

``aceverify.train.train_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/train.py:69``

.. code:: python

   train_model(config)

+-----------------------+----------------------------+------------------------------------------------------+
| Config Key            | Type                       | Description                                          |
+=======================+============================+======================================================+
| ``model``             | ``ACEVerifyModel``         | The model instance                                   |
+-----------------------+----------------------------+------------------------------------------------------+
| ``device``            | ``torch.device``           | Target device                                        |
+-----------------------+----------------------------+------------------------------------------------------+
| ``epochs``            | ``int``                    | Number of training epochs                            |
+-----------------------+----------------------------+------------------------------------------------------+
| ``criterion``         | ``nn.Module``              | Loss function                                        |
+-----------------------+----------------------------+------------------------------------------------------+
| ``optimizer``         | ``optim.Optimizer``        | Optimizer                                            |
+-----------------------+----------------------------+------------------------------------------------------+
| ``scheduler``         | ``lr_scheduler.Scheduler`` | Learning rate scheduler                              |
+-----------------------+----------------------------+------------------------------------------------------+
| ``train_path``        | ``str``                    | HDF5 training file path                              |
+-----------------------+----------------------------+------------------------------------------------------+
| ``test_path``         | ``str``                    | HDF5 test file path                                  |
+-----------------------+----------------------------+------------------------------------------------------+
| ``batch_size``        | ``int``                    | Data loader batch size                               |
+-----------------------+----------------------------+------------------------------------------------------+
| ``checkpoint_path``   | ``str``                    | Where to save the checkpoint                         |
+-----------------------+----------------------------+------------------------------------------------------+
| ``shuffle_data``      | ``bool``                   | Whether to shuffle training data (default: ``True``) |
+-----------------------+----------------------------+------------------------------------------------------+
| ``train_indices``     | ``list[int]``              | Pre-selected training indices (default: ``[]``)      |
+-----------------------+----------------------------+------------------------------------------------------+
| ``test_indices``      | ``list[int]``              | Pre-selected test indices (default: ``[]``)          |
+-----------------------+----------------------------+------------------------------------------------------+
| ``dataset_class``     | ``type``                   | Dataset class to use (default: ``ACEDataset``)       |
+-----------------------+----------------------------+------------------------------------------------------+

**Returns**: ``(model, metrics)`` where ``metrics`` is a dict with keys: ``epochs``, ``train_accuracies``, ``test_accuracies``.

--------------

``aceverify.train.load_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/train.py:27``

.. code:: python

   load_data(indices, path, n, training, shuffle_data=True, dataset_class=ACEDataset)

+-------------------+-----------------+-----------------+-------------------------------------------+
| Parameter         | Type            | Default         | Description                               |
+===================+=================+=================+===========================================+
| ``indices``       | ``list[int]``   | *required*      | Sample indices (empty list = auto-select) |
+-------------------+-----------------+-----------------+-------------------------------------------+
| ``path``          | ``str``         | *required*      | HDF5 file path                            |
+-------------------+-----------------+-----------------+-------------------------------------------+
| ``n``             | ``int``         | *required*      | Total number of samples to select         |
+-------------------+-----------------+-----------------+-------------------------------------------+
| ``training``      | ``bool``        | *required*      | Whether to apply augmentation             |
+-------------------+-----------------+-----------------+-------------------------------------------+
| ``shuffle_data``  | ``bool``        | ``True``        | Whether to shuffle selected indices       |
+-------------------+-----------------+-----------------+-------------------------------------------+
| ``dataset_class`` | ``type``        | ``ACEDataset``  | Dataset class to instantiate              |
+-------------------+-----------------+-----------------+-------------------------------------------+

**Returns**: An ``ACEDataset`` (or ``dataset_class``) instance.

**Behavior**: If ``indices`` is empty, reads all labels from the HDF5 file, identifies real and fake sample indices, and randomly selects ``n // 2`` from each class (balanced sampling).

--------------

``aceverify.preprocess.preprocess_dataset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/preprocess.py:166``

.. code:: python

   preprocess_dataset(zip_file_path, subfolder, h5_path, temp_dir=None, ffmpeg_bin=None)

+-------------------+-------------------------+----------------------------------------+------------------------------------+
| Parameter         | Type                    | Default                                | Description                        |
+===================+=========================+========================================+====================================+
| ``zip_file_path`` | ``str``                 | *required*                             | Path to the DFDC-style zip archive |
+-------------------+-------------------------+----------------------------------------+------------------------------------+
| ``subfolder``     | ``str``                 | *required*                             | Subfolder name within the zip      |
+-------------------+-------------------------+----------------------------------------+------------------------------------+
| ``h5_path``       | ``str``                 | *required*                             | Output HDF5 file path              |
+-------------------+-------------------------+----------------------------------------+------------------------------------+
| ``temp_dir``      | ``str \| Path \| None`` | ``None`` (defaults to ``<repo>/temp``) | Temporary extraction directory     |
+-------------------+-------------------------+----------------------------------------+------------------------------------+
| ``ffmpeg_bin``    | ``str \| None``         | ``None`` (resolves from env/PATH)      | ffmpeg executable path             |
+-------------------+-------------------------+----------------------------------------+------------------------------------+

--------------

``aceverify.preprocess.process_vid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/preprocess.py:51``

.. code:: python

   process_vid(file_name, subfolder, temp_dir, ffmpeg_bin)

**Returns**: ``True`` if face detection and frame processing succeeded, ``False`` otherwise.

--------------

``aceverify.preprocess.save_vid_to_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/preprocess.py:104``

.. code:: python

   save_vid_to_h5(file_name, label, h5_path, temp_dir)

Saves the 16 processed frames and audio clip to the HDF5 file under a new group. If a group with the same name already exists, it is deleted first.

--------------

``aceverify.visualize_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/visualize_data.py``

+------------------------+------------------------+----------------------------------------------------------------------+
| Function               | Signature              | Description                                                          |
+========================+========================+======================================================================+
| ``test_visualization`` | ``(dataset, index=0)`` | Displays 4 sampled frames and the Mel-spectrogram using matplotlib   |
+------------------------+------------------------+----------------------------------------------------------------------+
| ``numRealAndFake``     | ``(dataset)``          | Counts and prints the number of real and fake samples in the dataset |
+------------------------+------------------------+----------------------------------------------------------------------+

--------------

Data Structures
---------------

HDF5 File Format
~~~~~~~~~~~~~~~~

Each HDF5 file contains multiple top-level groups, one per video sample:

::

   /
   ├── {video_basename}/
   │   ├── video           # numpy.ndarray [16, 224, 224, 3] uint8, gzip-compressed
   │   ├── audio           # numpy.ndarray [N] float32, gzip-compressed
   │   └── attrs:
   │       └── label       # int: 0 (Real) or 1 (Fake)
   ├── {next_video}/
   │   └── ...

Dataset Specifications
^^^^^^^^^^^^^^^^^^^^^^

+-------------+-----------------------+-------------+-------------+---------------------------------------------------------------+
| Dataset     | Shape                 | Dtype       | Compression | Description                                                   |
+=============+=======================+=============+=============+===============================================================+
| ``video``   | ``(16, 224, 224, 3)`` | ``uint8``   | ``gzip``    | 16 processed face-cropped frames at 224x224 resolution        |
+-------------+-----------------------+-------------+-------------+---------------------------------------------------------------+
| ``audio``   | ``(N,)``              | ``float32`` | ``gzip``    | Raw audio waveform samples (0.5s clip extracted at 5s offset) |
+-------------+-----------------------+-------------+-------------+---------------------------------------------------------------+

Attribute Specifications
^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------+-----------------+-----------------+--------------------------------------------------+
| Attribute       | Type            | Value           | Description                                      |
+=================+=================+=================+==================================================+
| ``label``       | ``int``         | ``0`` or ``1``  | ``0`` = Real/Authentic, ``1`` = Fake/Manipulated |
+-----------------+-----------------+-----------------+--------------------------------------------------+

DFDC JSON Label Format
~~~~~~~~~~~~~~~~~~~~~~

The preprocessing pipeline expects a JSON file within the zip archive that maps video filenames to their labels:

.. code:: json

   {
       "video_file_1.mp4": {"label": "FAKE"},
       "video_file_2.mp4": {"label": "REAL"},
       "video_file_3.mp4": {"label": "FAKE"}
   }

Label Mapping
~~~~~~~~~~~~~

============ ============= ==========================
String Label Integer Label Description
============ ============= ==========================
``"REAL"``   ``0``         Authentic media
``"FAKE"``   ``1``         Manipulated/deepfake media
============ ============= ==========================

--------------

Model Class Reference
---------------------

``ACEVerifyModel``
~~~~~~~~~~~~~~~~~~

   **Source**: ``aceverify/model.py:43``

Full multimodal deepfake detection model. See the `Architecture and Pipeline <Architecture-and-Pipeline>`__ page for detailed architecture documentation.

``DeepfakeEfficientNet``
~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``models/efficientnet.py:5``

.. code:: python

   class DeepfakeEfficientNet(nn.Module):
       def __init__(self):
           self.model = timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=1)
       def forward(self, x):
           return self.model(x)

A lightweight 2D image classifier wrapping the EfficientNet-B4 (Noisy Student) backbone with a single binary classification head. Input: ``[B, 3, 224, 224]``. Output: ``[B, 1]`` raw logit.

``DeepfakeXception``
~~~~~~~~~~~~~~~~~~~~

   **Source**: ``models/xception.py:4``

.. code:: python

   class DeepfakeXception(nn.Module):
       def __init__(self):
           self.model = timm.create_model('xception', pretrained=True, num_classes=1)
       def forward(self, x):
           return self.model(x)

A 2D image classifier wrapping the XceptionNet backbone. Input: ``[B, 3, 224, 224]``. Output: ``[B, 1]`` raw logit.

``ACEVerifyIntegration``
~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``models/ace_verify.py:3``

.. code:: python

   class ACEVerifyIntegration(aceverify.model.ACEVerifyModel):
       def __init__(self):
           super().__init__()
           # Load the model weights (JIT serialized) from the final training checkpoint

Extends ``ACEVerifyModel`` to support loading JIT-serialized (TorchScript) model weights.

--------------

Evaluation Script Reference
---------------------------

``evaluation.evaluate.evaluate()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``evaluation/evaluate.py:20``

.. code:: python

   def evaluate(h5_path, checkpoint_path, batch_size=8, device=None):

Runs batch inference on an HDF5 test set and saves predictions to CSV.

+---------------------+--------------------------+------------------------+------------------------------+
| Parameter           | Type                     | Default                | Description                  |
+=====================+==========================+========================+==============================+
| ``h5_path``         | ``str``                  | *required*             | Path to the test HDF5 file   |
+---------------------+--------------------------+------------------------+------------------------------+
| ``checkpoint_path`` | ``str``                  | *required*             | Path to the model checkpoint |
+---------------------+--------------------------+------------------------+------------------------------+
| ``batch_size``      | ``int``                  | ``8``                  | Inference batch size         |
+---------------------+--------------------------+------------------------+------------------------------+
| ``device``          | ``torch.device \| None`` | ``None`` (auto-detect) | Target device                |
+---------------------+--------------------------+------------------------+------------------------------+

**Returns**: A ``pandas.DataFrame`` with columns ``label`` and ``pred``.

``evaluation.aceverify_test.evaluate_veriface()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``evaluation/aceverify_test.py:36``

.. code:: python

   def evaluate_veriface(dataloader, model, device):

Runs inference for the ACEVerifyModel on a data loader.

**Returns**: ``(all_labels, all_preds)`` where each is a list of integers.

``evaluation.spatial2D_test.evaluate_spatial_baseline()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``evaluation/spatial2D_test.py:39``

.. code:: python

   def evaluate_spatial_baseline(test_loader, model, device):

Runs inference for 2D spatial baseline models (EfficientNet, Xception). Reshapes video frames ``[B, C, T, H, W]`` to ``[B*T, C, H, W]`` for per-frame classification, then aggregates predictions by averaging per-frame fake probabilities.

**Returns**: ``(all_labels, all_preds)`` where each is a list.

``evaluation.timeSformer_test.evaluate()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``evaluation/timeSformer_test.py:37``

.. code:: python

   def evaluate(dataloader, model, device):

Runs inference for the TimeSformer model (``facebook/timesformer-base-finetuned-k400``). Requires the ``transformers`` library.

**Returns**: ``(all_labels, all_preds)`` where each is a list.

--------------

Progress Streaming
------------------

   **Source**: ``frontend/app.py:240``

The Streamlit web application uses a single ``st.progress`` bar that is updated in place across all analysis stages:

.. code:: python

   progress_bar = st.progress(0, text=f"Loading {model_choice}…")

   # Stage 1: Frame extraction (0% → 25%)
   progress_bar.progress(10, text="Extracting & Processing frames…")
   # ... process video ...
   progress_bar.progress(25, text="Running inference…")

   # Stage 2: Grad-CAM generation (25% → 50%)
   progress_bar.progress(50, text="Generating Grad-CAM…")

   # Stage 3: Timeline scoring (50% → 100%)
   progress_bar.progress(75, text="Scoring timeline…")
   progress_bar.progress(90, text="Scoring timeline…")
   progress_bar.progress(100, text="Done ✦")
   progress_bar.empty()

Progress Budget
~~~~~~~~~~~~~~~

+-----------------------+-----------------------+--------------------------------------------+
| Stage                 | Progress Range        | Description                                |
+=======================+=======================+============================================+
| Model loading         | 0%                    | Load the selected model checkpoint         |
+-----------------------+-----------------------+--------------------------------------------+
| Frame extraction      | 0% – 25%              | Extract frames via ``FaceProcessor``       |
+-----------------------+-----------------------+--------------------------------------------+
| Inference             | 25% – 50%             | Run model forward pass                     |
+-----------------------+-----------------------+--------------------------------------------+
| Grad-CAM              | 50% – 75%             | Generate attention heatmap overlay         |
+-----------------------+-----------------------+--------------------------------------------+
| Timeline scoring      | 75% – 100%            | Compute temporal scores and evidence flags |
+-----------------------+-----------------------+--------------------------------------------+

The bar is emptied (``progress_bar.empty()``) after completion to remove it from the UI.

State Update Pattern
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   if analyze_clicked and not st.session_state.analyzed:
       # ... run analysis ...
       st.session_state.results = { ... }
       st.session_state.analyzed = True

Analysis results are stored in ``st.session_state.results`` and the ``analyzed`` flag is set to ``True``. This triggers the results section rendering on the next rerun. The ``not st.session_state.analyzed`` guard prevents re-running analysis if the user changes options without clicking “Analyze” again.

--------------

Environment Configuration
-------------------------

Conda Environments
~~~~~~~~~~~~~~~~~~

===================== ============== ===================================
File                  Python Version Purpose
===================== ============== ===================================
``conda_env.yml``     3.13           Minimal conda base environment
``conda_env_new.yml`` 3.11           Full ML stack with pip dependencies
===================== ============== ===================================

Streamlit Configuration
~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``.streamlit/config.toml``

.. code:: toml

   [server]
   enableStaticServing = true

Enables Streamlit’s static file serving, which is required by the upload-card media preview for low-latency same-origin playback of preset videos and user uploads.

Docker Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Source**: ``Dockerfile:3``

+-----------------------------+-----------------------+------------------------------------------------+
| Variable                    | Value                 | Purpose                                        |
+=============================+=======================+================================================+
| ``DEBIAN_FRONTEND``         | ``noninteractive``    | Suppress apt-get prompts                       |
+-----------------------------+-----------------------+------------------------------------------------+
| ``PYTHONDONTWRITEBYTECODE`` | ``1``                 | Prevent .pyc file generation                   |
+-----------------------------+-----------------------+------------------------------------------------+
| ``PYTHONUNBUFFERED``        | ``1``                 | Enable unbuffered output for real-time logging |
+-----------------------------+-----------------------+------------------------------------------------+
| ``PIP_NO_CACHE_DIR``        | ``1``                 | Disable pip cache to reduce image size         |
+-----------------------------+-----------------------+------------------------------------------------+
| ``PYTHONPATH``              | ``/workspace``        | Enable module imports inside the container     |
+-----------------------------+-----------------------+------------------------------------------------+

Runtime Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------+-------------------------------------------+----------------------------------+
| Variable                              | Purpose                                   | Default                          |
+=======================================+===========================================+==================================+
| ``FFMPEG_BIN``                        | Override path to ffmpeg executable        | Resolved from ``PATH``           |
+---------------------------------------+-------------------------------------------+----------------------------------+
| ``CUDA_VISIBLE_DEVICES``              | Restrict GPU visibility                   | All GPUs                         |
+---------------------------------------+-------------------------------------------+----------------------------------+
| ``FFMPEG_BIN`` (in ``preprocess.py``) | Fallback if ``--ffmpeg-bin`` not provided | ``os.environ.get('FFMPEG_BIN')`` |
+---------------------------------------+-------------------------------------------+----------------------------------+
