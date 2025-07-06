<p align="center">
    <img src="assets\dynamicvoyager_icon2.jpg" width="20%">
</p>
<div align="center">

# ✨Voyaing into Unbounded Dynamic Scenes from a Single View

<p align="center">
<a href="https://tianfr.github.io/">Fengrui Tian</a>,
<a href="https://tianjiaoding.com/">Tianjiao Ding</a>,
<a href="https://peterljq.github.io/">Jinqi Luo</a>,
<a href="https://hanchmin.github.io/">Hancheng Min</a>,
<a href="http://vision.jhu.edu/rvidal.html">René Vidal</a>
<br>
    University of Pennsylvania
</p>
<h3 align="center">🌟ICCV 2025🌟</h3>
<!-- <a href=""><img src='https://img.shields.io/badge/arXiv-2507.02813-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp; -->
<a href="https://tianfr.github.io/project/DynamicVoyager/index.html"><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets\dynamicvoyager_teaser.gif" alt="gif1" style="flex: 1 1 20%; max-width: 98%;">
</div>

<table>
  <tr>
    <td><img src="assets\village.gif" alt="gif5" width="150"></td>
    <td><img src="assets\rose.gif" alt="gif1" width="150"></td>
    <td><img src="assets\umbrella.gif" alt="gif2" width="150"></td>
    <td><img src="assets\village1.gif" alt="gif3" width="150"></td>
    <td><img src="assets\cat.gif" alt="gif4" width="150"></td>
    <td><img src="assets\cartoon.gif" alt="gif5" width="150"></td>
    <td><img src="assets\village2.gif" alt="gif5" width="150"></td>
  </tr>
</table>



This is the official implementation of our ICCV 2025 paper "Voyaging into Unbounded Dynamic Scenes from a Single View".


## Abstract
 We study the problem of generating an unbounded dynamic scene from a single view. Since the scene is changing over time, different generated views need to be consistent with the underlying 3D motions. We propose DynamicVoyager that reformulates the dynamic scene generation as a scene outpainting process for new dynamic content. As 2D outpainting models can hardly generate 3D consistent motions from only 2D pixels at a single view, we consider pixels as rays to enrich the pixel input with the ray context, so that the 3D motion consistency can be learned from the ray information. More specifically, we first map the single-view video input to a dynamic point cloud with the estimated video depths. Then we render the partial video at a novel view and outpaint the video with ray contexts from the point cloud to generate 3D consistent motions. We employ the outpainted video to update the point cloud, which is used for scene outpainting from future novel views.

🎮 Codes, 📖 paper and 🪄 models will be released soon....



## Citation
```
@InProceedings{25iccv/tian_dynvoyager,
    author    = {Tian, Fengrui and Ding, Tianjiao and Luo, Jinqi and Min, Hancheng and Vidal, Ren\'e},
    title     = {Voyaging into Unbounded Dynamic Scenes from a Single View},
    booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025}
}
```

## Contact
If you have any questions, please feel free to contact [Fengrui Tian](https://tianfr.github.io).
