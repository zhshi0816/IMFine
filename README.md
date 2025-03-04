# IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement

This repo is the official implementation of 'IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement', CVPR 2025.

[Paper](), [Project Page](https://xinxinzuo2353.github.io/imfine/), [Evaluation Results](https://drive.google.com/file/d/1r5KytaSENNRIObyVbnDcAVAnhPHd7490/view?usp=drive_link), [Dataset](https://drive.google.com/file/d/1DLuk9KiHPhlK9QpJZthUmIhDQIZH_cOD/view?usp=drive_link)

## Packages
The following pakages are required to run the code:
* python==3.9.8
* pytorch==2.2.0
* cudatoolkit==11.8
* torchvision==0.17.0
* opencv-python==4.11.0


## Evaulation
* You can evaluate our results by:
```
python evaluation.py
```

* changing `outputs` in `evaluation.py` to evaluate yours.

## Main Algorithm
Due to IP-policy, we don't have plan to release the codes at the moment.

## Citation
Please consider citing this paper if you find the code and data useful in your research:
```
@inproceedings{imfine,
    title={IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement}, 
    author={Shi, Zhihao and Huo, Dong and Zhou, Yuhongze and Yin, Kejia and Min, Yan and Lu, Juwei and Zuo, Xinxin},
    year={2025},
    booktitle={CVPR},
}
```


## References
Some other great 3D Inpainting resources that we benefit from:
* [SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields](https://arxiv.org/abs/2211.12254)
* [MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior](https://arxiv.org/abs/2405.02859)
