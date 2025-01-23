KARMA: Knowledge Distillation to RBF-based Meta-Learner for Industrial Anomaly Detection
-------------
<span style="color:black;"> ***KARMA*** is a lightweight anomaly detection model, which utilizes ensemble learning and knowledge distillation for Industrial Control Systems (ICS). The model includes prediction values from ten state-of-the-art anomaly detection models that have been evaluated on two public datasets. </span>

This repository contains the ***KARMA*** implemented in python and the dataset.

Getting started
-------------
To access the source code, clone this repository using the following command.

**Clone the repo**

```bash
git clone https://github.com/anonymous-code-dev/anonymous.git && cd KARMA 
```

**Dataset**

The datasets utilized in this paper can be found in the ``datasets`` folder.

Build environment
-------------
<span style="color:black;"> Our code is run in the following environment as follows. </span>

### Requirements

```
Pytorch: 2.2.1 
TensorFlow: 2.11.0
Python: 3.8.1
```

### Install packages

```bash
pip install -r requirements.txt
```

Run
-------------
Once the repository and environment settings are complete, ***KARMA*** can run as following commands:  
(If you want to see the code result directly, please check <code>scripts/KARMA_SWaT.py</code> and <code>scripts/KARMA_WADI.py</code>)  

<pre><code>#run on SWaT 
python KARMA_SWaT.py
  
#run on WADI 
python KARMA_WADI.py</code></pre>


