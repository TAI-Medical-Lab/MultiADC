


<!-- PROJECT SHIELDS -->



<!-- PROJECT LOGO -->
<br />

<p align="center">

  <h1 align="center">MultiADC: Advanced Antibody-Drug Conjugate
Activity Prediction through Multi-scale Feature
Fusion</h1>

</p>

### Requirements Before Development
1. **Python Version**: Python 3.9.18
2. **Graphics Card Support**: At least 4GB of CUDA memory
3. **Install Python Packages**:
```sh
pip install -r requirements
```
### Getting Started Guide
- **Dataset Storage Path**: `/dataset`
Generate molecular DGL graphs using `smile_process.py` and ESM embedding features using `protein_process.py`.

If you want to perform data augmentation, you need to execute `augment_process` first.

If you need to modify the model structure, you can make changes in `model/net.py`.

Start your training by running `python main.py`.

- **Start Training**:
```sh
python main.py
```
 



