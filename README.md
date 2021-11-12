# statistical-learning-project-part2 

## Summary of Project Tasks:

The objective is to study how to use a common SVM package by implementing some classification
tasks.

Data Set: The given dataset contains 50 categories/classes. The training set has 4786 samples in the file
‘trainData.mat’, and the testing set has 1833 samples in the file ‘testData.mat’. Each sample is described
by the rows of 3 different feature matrices i.e., X1 ,X2, and X3 in the corresponding file, and the category
vector is always Y. All the 3 features are normalized histograms, which means the elements are non-
negative and the sum of each feature equals to 1 (i.e.,**∑**<sub>𝑗</sub> 𝑿<sub>𝑘</sub>(𝑖, 𝑗) ≡ 1).

Step 0: Classification by individual features.
Output: The classification accuracy for the testing set in the follow cases (1) and (2).

Instructions:
(1) For each of the 3 features in the training set, 𝑿<sub>𝑘</sub> (1 ≤ 𝑘 ≤ 3), train a multi-class linear SVM classifier,
i.e., **h**<sub>𝑘</sub>(𝐱). Get the prediction result of **h**<sub>𝑘</sub>(𝐱) based on the same feature 𝑿<sub>𝑘</sub> in the testing set and compare to 𝒀 for computing the classification accuracy.
(2) Based on the SVM classifiers **h**<sub>𝑘</sub>(𝐱), we can also obtain 𝑝<sub>𝑘</sub>(𝑤<sub>𝑖</sub>|𝐱), the (posterior) probability of sample 𝐱 that it belongs to the 𝑖-th category (𝑤<sub>𝑖</sub>) according to feature 𝑿<sub>𝑘</sub> (1 ≤ 𝑘 ≤ 3). This can be done by using the parameter ‘-b 1’ option in training and testing (check http://www.csie.ntu.edu.tw/~cjlin/libsvm/ for more details). Train the SVM classifiers with this option and report the classification accuracies on the testing set based on the 3 features respectively.

Step 1: Feature combination by fusion of classifiers.
Output: The classification accuracy in the testing set and compare it to that of (2) in Step 0.

Instructions: Directly combine the 3 SVM classifiers with probability output i.e., 𝑝<sub>𝑘</sub>(𝑤<sub>𝑖</sub>|𝐱) (1 ≤ 𝑘 ≤ 3), in (2) of Step 0. Combine the 3 classifiers by probability fusion as 𝑝<sub>𝑘</sub>(𝑤<sub>𝑖</sub>|𝐱) = **∑**<sub>𝑘</sub> 𝑝<sub>𝑘</sub>(𝑤<sub>𝑖</sub>|𝐱)⁄3 . The final recognition result is 𝑤<sub>𝑖*</sub> = argmax<sub>𝑖</sub> 𝑝<sub>𝑘</sub>(𝑤<sub>𝑖</sub>|𝐱).

Step 2: Feature combination by simple concatenation.
Output: The classification accuracy in the testing set and compare it to that of (1) in Step 0.

Instructions: Directly concatenate the 3 features 𝐗<sub>𝑘</sub> , 1 ≤ 𝑘 ≤ 3 to form a single feature, i.e. 𝐗 = [𝐗<sub>1</sub>, . . ., 𝐗<sub>k</sub>] train a linear SVM classifier based on 𝐗 and obtain the classification accuracy for the testing set.
