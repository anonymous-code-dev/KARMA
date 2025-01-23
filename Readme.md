KARMA: Knowledge Distillation to RBF-based Meta-Learner for Industrial Anomaly Detection
-------------
<span style="color:black;"> ***KARMA*** is a lightweight anomaly detection model, which utilizes ensemble learning and knowledge distillation for Industrial Control Systems (ICS). The model includes prediction values from ten state-of-the-art anomaly detection models that have been evaluated on two public datasets. </span>

This repository contains the ***KARMA*** implemented in python and the dataset.

Getting started
-------------
We confirmed that ***KARMA*** runs on Ubuntu 20.04.

**Clone the repo**

* <span style="color:black;"> To access the source code, clone this repository using the following command: </span>

```bash
<pre><code><span style="color:black;"> git clone https://github.com/anonymous-code-dev/anonymous.git && cd KARMA 
```

Build environment
-------------
<span style="color:black;"> Our implementation environment is as follows: </span>

* **Pytorch version 2.2.1**
  
* **TensorFlow 2.11.0**
  
* **Python 3.8.1**

Implementation
-------------
Once the repository and environment settings are complete, ***KARMA*** can run as following commands:  
(If you want to see the code result directly, please check <code>KARMA_SWaT.py</code> and <code>KARMA_WADI.py</code>.)  

<pre><code>#SWaT 
python KARMA_SWaT.py
  
#WADI 
python KARMA_WADI.py</code></pre>


