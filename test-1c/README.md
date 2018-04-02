This tests the subMMNIST dataset, which is the same thing as the MMNIST dataset, except it doesn't have any sixes.

We train a capsule network on the subMMNIST dataset for 10000 iterations, and then launch it on the entire MMNIST dataset, and see what happens.


| Step        | Accuracy (0px) |
| ------------- |:-------------:|
| 0 | 0.191 |
| 100 | 0.227 |