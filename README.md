<h1>Enhancing Image Classification with ConvNextV2 and Squeeze-and-Excitation Blocks: A Comparative Study on Mixed Datasets</h1>

<h2>Overview</h2>
<p>This project focuses on the foundational task of image classification within computer vision, aiming to categorize pixels in images into one or more classes. We utilize Convolutional Neural Networks (CNNs), particularly the advanced ConvNextV2 architecture, to classify images from three datasets: CUB birds, FGVC aircraft, and FoodX.</p>

<h2>Contributions</h2>
<ul>
  <li>Develop a robust and accurate image classification model.</li>
  <li>Explore various techniques to enhance model accuracy across multiple datasets.</li>
  <li>Investigate the effects of architecture modifications, layer freezing, dropout, attention mechanisms, loss functions, and hyperparameters.</li>
</ul>

<h2>Architecture</h2>
<p>The architecture is based on the ConvNextV2 large variant, incorporating custom modifications for enhanced feature recalibration and attention to relevant features.</p>
<p align="center">
  <img src="https://github.com/Sarim-MBZUAI/Enhanced-Image-Classification-ConvNextV2-and-SE-Blocks-Study/blob/main/ConvNextL%2BAtt12-1.png" alt="our Architecture" width="75%"/>
  <br>
  <strong>Figure 1:</strong> Our Architecture
</p>


<h3>Modifications</h3>
<ul>
  <li><strong>Squeeze-and-Excitation (SE) Blocks</strong>: Enhance feature recalibration.</li>
  <li><strong>ModifiedBlock Module</strong>: Integrates SE blocks into the architecture seamlessly.</li>
</ul>

<h2>Implementation Details</h2>
<p>Custom dataset classes are created for each dataset, utilizing PyTorch’s dataset handling capabilities to ensure efficient data loading.</p>

<h3>Training Specifications</h3>
<ul>
  <li><strong>Input Dimensions</strong>: 256 × 256</li>
  <li><strong>Batch Size</strong>: 16</li>
  <li><strong>Optimizer</strong>: Adam with StepLR and CosineAnnealingLR for learning rate adjustments.</li>
</ul>

<h2>Results and Performance</h2>
<p>Enhancements in the model architecture have shown significant improvements in accuracy.</p>

<h3>Performance Comparison</h3>
<table>
  <tr>
    <th>Model</th>
    <th>Dataset Combination</th>
    <th>Test Accuracy (%)</th>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>CUB birds</td>
    <td>89.42</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>CUB birds + FGVC Aircraft</td>
    <td>88.45</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>FoodX</td>
    <td>78.00</td>
  </tr>
  <tr>
    <td>Baseline + SE block</td>
    <td>CUB birds</td>
    <td>89.96</td>
  </tr>
  <tr>
    <td>Baseline + SE block</td>
    <td>CUB birds + FGVC Aircraft</td>
    <td>89.95</td>
  </tr>
  <tr>
    <td>Baseline + SE block</td>
    <td>FoodX</td>
    <td>79.40</td>
  </tr>
</table>

<h3>Ablation Study: Frozen Stages Accuracy</h3>
<table>
  <tr>
    <th>Frozen Stages</th>
    <th>Accuracy (%)</th>
  </tr>
  <tr>
    <td>Stage 1</td>
    <td>89.52</td>
  </tr>
  <tr>
    <td>Stage 1 and 2</td>
    <td>88.99</td>
  </tr>
  <tr>
    <td>Stage 1, 2, and 3</td>
    <td>80.31</td>
  </tr>
  <tr>
    <td>All Stages</td>
    <td>52.31</td>
  </tr>
</table>

<h3>Ablation Study: SE Block Positioning</h3>
<table>
  <tr>
    <th>Position</th>
    <th>Accuracy (%)</th>
  </tr>
  <tr>
    <td>Between Stage 1 and 2</td>
    <td>89.93</td>
  </tr>
  <tr>
    <td>Between Stage 2 and 3</td>
    <td>89.65</td>
  </tr>
  <tr>
    <td>Between Stage 3 and 4</td>
    <td>89.361</td>
  </tr>
</table>

<h2>Discussion</h2>
<p>The experimental results highlight the effectiveness of SE blocks and the strategic freezing of layers in enhancing model performance, focusing on the most informative features and improving accuracy across datasets.</p>
