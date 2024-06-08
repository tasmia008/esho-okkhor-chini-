<!DOCTYPE html>
<html>
<head>
  <title>Bangla Handwritten Character Recognition</title>
</head>
<body>

<h1>Esho_Okkhor-Chini</h1>

<h2>Overview</h2>
<p>This project focuses on recognizing Bangla handwritten characters using a Convolutional Neural Network (CNN) implemented with PyTorch. The model is trained on a dataset of handwritten characters, and various techniques are used for data preprocessing, model training, and evaluation. Additionally, LIME (Local Interpretable Model-agnostic Explanations) is utilized to interpret the model’s predictions.</p>

<h2>Features</h2>
<ul>
  <li>Data loading and preprocessing</li>
  <li>CNN model implementation and training</li>
  <li>Model evaluation and testing</li>
  <li>Interpretability of model predictions using LIME</li>
</ul>

<h2>Dataset</h2>
<p>The dataset for this project is stored on Google Drive and includes images of Bangla handwritten characters. The dataset is downloaded and unzipped as part of the preprocessing steps.</p>

<h2>Installation</h2>
<p>To run this project, you'll need to install the following dependencies:</p>
<pre><code>pip install torch torchvision pandas numpy matplotlib lime</code></pre>

<h2>Usage</h2>
<ol>
  <li><strong>Mount Google Drive:</strong>
    <pre><code>from google.colab import drive
drive.mount('/content/gdrive')</code></pre>
  </li>
  <li><strong>Copy and Unzip Dataset:</strong>
    <pre><code>!cp /content/gdrive/MyDrive/combine_data.zip ./
!unzip combine_data.zip</code></pre>
  </li>
  <li><strong>Preprocess Data:</strong>
    <p>The script processes the dataset and creates labeled data files.</p>
  </li>
  <li><strong>Train the Model:</strong>
    <p>The CNN model is implemented and trained using PyTorch. Training involves loading the data, defining the network architecture, specifying the loss function and optimizer, and iterating through the training process.</p>
  </li>
  <li><strong>Evaluate the Model:</strong>
    <p>Evaluate the trained model on a test dataset to measure its accuracy and reliability.</p>
  </li>
  <li><strong>Interpret the Model:</strong>
    <p>Use LIME to interpret the model’s predictions and understand the decision-making process.</p>
  </li>
</ol>

<h2>Project Structure</h2>
<ul>
  <li><code>data/</code>: Directory containing the dataset.</li>
  <li><code>src/</code>: Source code for data preprocessing, model training, and evaluation.</li>
  <li><code>notebooks/</code>: Jupyter notebooks for exploratory data analysis and experiments.</li>
  <li><code>models/</code>: Saved models and checkpoints.</li>
</ul>

<h2>Key Files</h2>
<ul>
  <li><code>data_preprocessing.py</code>: Script for preprocessing the data.</li>
  <li><code>train_model.py</code>: Script for training the CNN model.</li>
  <li><code>evaluate_model.py</code>: Script for evaluating the model.</li>
  <li><code>interpret_model.py</code>: Script for interpreting the model using LIME.</li>
  <li><code>bangla_handwritten_character_recognation.ipynb</code>: Jupyter notebook for the entire workflow.</li>
</ul>

<h2>Results</h2>
<p>The trained model achieved high accuracy on the test dataset, demonstrating its effectiveness in recognizing Bangla handwritten characters. The use of LIME provided valuable insights into the model’s decision-making process, highlighting the areas of the image that contributed most to the predictions.</p>

<h2>Contributing</h2>
<p>If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more details.</p>

<h2>Acknowledgements</h2>
<p>The dataset used in this project is provided by <a href="https://rabby.dev/ekush/#home">source</a>. The LIME library was used for model interpretability.</p>

</body>
</html>
