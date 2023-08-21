# Concept_ZSL

## Pipeline

We pass the image through a standard backbone (Resnet for example), but instead of outputting a vector of size (1, num_classes), it's (1, num_features). The vector is then passed to the binary quantization layer which gives back a (1, num_features) vector with only 0s and 1s.

All we have to do then is multiply the output by the predicates matrix element-wise and substract back the output. OUT * predicates => AND operation (since they're binary). If OUT * predicates == predicates[label], then the sum of (OUT * predicate - OUT)[label] == 0, or else it will be some negative value. So the closest subset will ne the biggest value, thus CrossEntropy is usable.

![Pipeline](https://github.com/Tankiit/Concept_ZSL/blob/main/BinaryQuantizationSubset.png)