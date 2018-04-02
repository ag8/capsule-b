This tests the subMMNIST dataset, which is the same thing as the MMNIST dataset, except it doesn't have any sixes.

We train a capsule network on the subMMNIST dataset for 10000 iterations, and then launch it on the entire MMNIST dataset, and see what happens.


| Step        | Average accuracy (0px) | Caps bench (0px) | Conv bench (0px) |
| ------------- |:-------------:|:-------------:|:-------------:|
| 0 | 0.191 | 0.245 | 0.199 |
| 100 | 0.227 | 0.245 | |
| 200 | 0.225 | | |
| 300 | 0.323 | | |
| 400 | 0.395 | | |
| 500 | 0.449 | | |
| 600 | 0.486 | | |