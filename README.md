# VOKE
Code accompanying the paper Vision-Based Online Key Point Estimation of Deformable Robots.


<img src="figures/readme_images/fig1.png" alt="ViSE Pipeline" width="540">

## Performance
Estimation errors **with respect to the corresponding robot's length** 

|                                   | Estimation Technique.        | Number of Cameras | Robot Type  | Robot Length (mm) | Tip Error     |
| --------------------------------: | :--------------------------: | :---------------: | :---------: | :---------------: | :-----------: |
| **ViSE (ours)**                   | CNN                          | 2                 | WaxCast arm | 335               | **0.3%±0.2%** |
| **ViSE (ours)**                   | CNN                          | 2                 | SoPrA       | 270               | **0.5%±0.4%** |
| **ViSE (ours)**                   | CNN                          | 2                 | Soft fish   | 115               | **0.6%±0.6%** |
| [Camarillo et al.](https://ieeexplore.ieee.org/abstract/document/4543656)                  | 2D point-cloud fit           | 3                 | Soft arm    | 160               | 4.8%          |
| [Vandini et al.](https://ieeexplore.ieee.org/abstract/document/7913726)                    | Line feature detector        | 1                 | Soft arm    | 260               | 2.8%          |
| [Pedari et al.](https://ieeexplore.ieee.org/abstract/document/9071908)                     | LED light placement          | 2                 | Soft arm    | 468\*               | 4.5%          |
| [AlBeladi et al.](https://ieeexplore.ieee.org/abstract/document/9560751)                   | Edge detection & curve fit   | 1                 | Soft arm    | 287               | 4.5%±3.1%     |

> \* not provided, calculated based on their estimation data

## Requirements and Installation
The repo was written using *Python 3.8* with [`conda`](https://github.com/JacopoPan/a-minimalist-guide#install-conda) on *Ubuntu 20.04*

## Datasets
For an easy start, you can download our processed dataset on three different types of soft robots from [Google Drive](https://drive.google.com/drive/folders/1Rl2JSU4wbkHnazaM6Fxq8iWYGbndd91i?usp=sharing)

