# HW2 Bone

### Required Packages
* [`os-path`]
* [`numpy`]
* [`torch`]
* [`scikit-learn`]

### Required Files
* [`test-in.txt`]
* [`train-io.txt`]

### Training and Evaluating
* I've included a trained model [`bone.pth`] in the Repo, if you'd like to train a new model you can move it or delete it.
* The included model was trained on a GPU so it may require a GPU to evaluate it, if it does throw an error you can just delete it and train a new one.
* Running [`bone.py`] will train the model if there isn't a stored model in the directory.
* Run [`bone.py`] a second time to run an evaluation on the validation set and to regenerate [`test-out.txt`].
