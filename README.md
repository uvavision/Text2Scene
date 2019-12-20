# [Text2Scene: Generating Compositional Scenes from Textual Descriptions ](https://arxiv.org/abs/1809.01110)
Fuwen Tan, Song Feng, Vicente Ordonez. CVPR 2019


## Overview
In this work, we propose Text2Scene, a model that generates various forms of compositional scene representations from natural language descriptions. Unlike recent works, our method does NOT use Generative Adversarial Networks (GANs). Text2Scene instead learns to sequentially generate objects and their attributes (location, size, appearance, etc) at every time step by attending to different parts of the input text and the current status of the generated scene. We show that under minor modifications, the proposed framework can handle the generation of different forms of scene representations, including cartoon-like scenes, object layouts corresponding to real images, and synthetic images. Our method is not only competitive when compared with state-of-the-art GAN-based methods using automatic metrics and superior based on human judgments but also has the advantage of producing interpretable results.

## Installation
- Setup a conda environment and install some prerequisite packages like this
```bash
conda create -n syn python=3.6          # Create a virtual environment
source activate syn         		# Activate virtual environment
conda install jupyter scikit-image cython opencv seaborn nltk pycairo   # Install dependencies
git clone https://github.com/cocodataset/cocoapi.git 			# Install pycocotools
cd cocoapi/PythonAPI
python setup.py build_ext install
python -m nltk.downloader all						# Install NLTK data
```
- Please also install [pytorch](http://pytorch.org/) 1.0 (or higher), torchVision, and torchtext
- Install the repo
```bash
git clone https://github.com/uvavision/Text2Scene.git
cd Text2Scene/lib
make
cd ..
```

## Data 
- Download the Abstract Scene and COCO datasets if you have not done so
```Shell
./experiments/scripts/fetch_data.sh
```
This will populate the `Text2Scene/data` folder with `AbstractScenes_v1.1`, `coco/images` and `coco/annotations`.
Please note that, for layout generation, we use coco2017 splits. But for composite image generation, we use coco2014 splits for fair comparisons with prior methods. The split info could be found in `Text2Scene/data/caches`.


## Demo
- Download the pretrained models
```Shell
./experiments/scripts/fetch_models.sh
```

- For the abstract scene and layout generation tasks, simply run
```Shell
./experiments/scripts/sample_abstract.sh	# Abstract Scene demo
./experiments/scripts/sample_layout.sh	# Layout demo
```
The scripts will take the example sentences in `Text2Scene/examples` as input. The step-by-step generation results will appear in `Text2Scene/logs`. Runing the scripts for the first time would be slow as it takes time to generate cache files (in `Text2Scene/data/caches`) for the datasets and download the GloVe data.

- To run the composite and inpainting demos, you need to download auxiliary data, including the image segment database and (optionally) the precomputed nearest neighbor tree. Be careful that the auxiliary data is around 30GB!!
```Shell
./experiments/scripts/fetch_aux.sh
./experiments/scripts/sample_composites.sh	 # Composites demo
./experiments/scripts/sample_inpainted.sh	 # Inpainting demo
```

Note that the demos will be run in CPU by default. To use GPU, simply add the `--cuda` flag in the scripts like:
```Shell
./tools/abstract_demo.py --cuda --pretrained=abstract_final
```

## Training
You can run the following scripts to train the models:
```Shell
./experiments/scripts/train_abstract.sh 		# Train the abstract scene model
./experiments/scripts/train_layout.sh 		# Train the coco layout model
./experiments/scripts/train_composites.sh 	# Train the composite image model
```
The composite image model will be trained using multiple GPUs by default. To use a single GPU, please remove the `--parallel` flag and modify the batch size using the `--batch_size` flag accordingly.

## TODO
I'm trying to also include the evaluation scripts and the generated images from the pretrained models to ease the evaluations.


## Citing

If you find our paper/code useful, please consider citing:

	@InProceedings{text2scene2019, 
	    author = {Tan, Fuwen and Feng, Song and Ordonez, Vicente},
	    title = {Text2Scene: Generating Compositional Scenes from Textual Descriptions},
	    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    month = {June},
	    year = {2019}}

    
# License

This project is licensed under the [MIT license](https://opensource.org/licenses/MIT):

Copyright (c) 2019 University of Virginia, Fuwen Tan, Song Feng, Vicente Ordonez.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







