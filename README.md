# QueryMatch

This is the official implementation of "QueryMatch: A Query-based Contrastive Learning Framework for Weakly Supervised Referring Image Segmentation". In this paper,we propose a novel one-stage weakly supervised RIS framework named QueryMatch, This framework reformulates RIS as a Query-Text matching problem. Furthermore, we propose a strategy, namely NSQE, to estimate the quality of negative samples. This strategy significantly boosts performance by selecting high-quality negative samples, emphasizing their uniqueness and difficulty in discrimination.

<p align="center">
	<img src="./figs/fig2.png" width="1000">
</p>


## Installation
- Clone this repo
```bash
git clone https://github.com/TensorThinker/QueryMatch.git
cd QueryMatch
```

- Create a conda virtual environment and activate it
```bash
conda create -n querymatch python=3.8 -y
conda activate querymatch
```

- Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/previous-versions)
- Install detectron following the [official installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
  
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

- Install apex following the [official installation guide](https://github.com/NVIDIA/apex)
- Compile the DCN layer:
  
```bash
cd utils_querymatch/DCN
./make.sh
```

```bash
cd mask2former
pip install -r requirements.txt
cd ./modeling/pixel_decoder/ops
sh make.sh
```

```bash
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
pip install albumentations
pip install Pillow==9.5.0
pip install tensorboardX
```

## Data Preparation

- Download images and Generate annotations according to [SimREC](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md).

- The project structure should look like the following:

```
| -- QueryMatch
     | -- data
        | -- anns
            | -- refcoco.json
            | -- refcoco+.json
            | -- refcocog.json
        | -- images
            | -- train2014
                | -- COCO_train2014_000000000072.jpg
                | -- ...
     | -- config_querymatch
     | -- configs
     | -- datasets
     | -- datasets_querymatch
     | -- DCNv2_latest
     | -- detectron2
     | -- mask2former
     | -- models_querymatch
     | -- ...
```
- NOTE: our Mask2former is trained on COCO’s training images, 
excluding those in RefCOCO, RefCOCO+, and RefCOCOg’s validation+testing. 

## QueryMatch

### Training
```
python train_querymatch.py --config ./config_querymatch/[DATASET_NAME].yaml --config-file ./configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml --eval-only MODEL.WEIGHTS [PATH_TO_MASK2FORMER_WEIGHT]

```

### Evaluation
```
python test_querymatch.py --config ./config_querymatch/[DATASET_NAME].yaml --eval-weights [PATH_TO_CHECKPOINT_FILE] --config-file ./configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml --eval-only MODEL.WEIGHTS [PATH_TO_MASK2FORMER_WEIGHT]

```

## Model Zoo

### QueryMatch
<table class="tg" style="undefined;table-layout: fixed">
<colgroup>
<col style="width: 140px">
<col style="width: 60px">
<col style="width: 60px">
<col style="width: 60px">
<col style="width: 60px">
<col style="width: 60px">
<col style="width: 60px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th class="tg-7btt"><span style="color:#000">Method</span></th>
    <th class="tg-7btt" colspan="3"><span style="color:#000">RefCOCO</span></th>
    <th class="tg-7btt" colspan="3"><span style="color:#000">RefCOCO+</span></th>
    <th class="tg-7btt"><span style="color:#000">RefCOCOg</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"><span style="color:#000">val</span></td>
    <td class="tg-c3ow"><span style="color:#000">testA</span></td>
    <td class="tg-c3ow"><span style="color:#000">testB</span></td>
    <td class="tg-c3ow"><span style="color:#000">val</span></td>
    <td class="tg-c3ow"><span style="color:#000">testA</span></td>
    <td class="tg-c3ow"><span style="color:#000">testB</span></td>
    <td class="tg-c3ow"><span style="color:#000">val-g</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"><strong>QueryMatch</td>
    <td class="tg-c3ow">57.82</td>
    <td class="tg-c3ow">56.54</td>
    <td class="tg-c3ow">58.43</td>
    <td class="tg-c3ow">37.88</td>
    <td class="tg-c3ow">38.35</td>
    <td class="tg-c3ow">37.33</td>
    <td class="tg-c3ow">37.85</td>
  </tr>
  <tr>
    <td class="tg-0pky"><strong>QueryMatch<sub>NSQE</sub></td>
    <td class="tg-c3ow">59.10</td>
    <td class="tg-c3ow">59.08</td>
    <td class="tg-c3ow">58.82</td>
    <td class="tg-c3ow">39.87</td>
    <td class="tg-c3ow">41.44</td>
    <td class="tg-c3ow">37.22</td>
    <td class="tg-c3ow">40.31</td>
  </tr>
</tbody>
</table>

## Notes
### Experimental Environment for Ours
- GPU: RTX 4090(24GB)
- CPU: 32 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- CUDA 11.7
### Compatibility Note
This project is compatible with multiple CUDA versions, including but not limited to CUDA 11.3. While the relative performance trends remain consistent across different hardware environments, please note that the specific numerical results may vary slightly.
## Acknowledgement

Thanks a lot for the nicely organized code from the following repos
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
