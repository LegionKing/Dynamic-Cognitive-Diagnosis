# Dynamic Cognitive Diagnosis

Source code  for the paper [Dynamic Cognitive Diagnosis: An Educational Priors-Enhanced Deep Knowledge Tracing Perspective](http://staff.ustc.edu.cn/~huangzhy/files/papers/FeiWang-TLT2023.pdf).



If this code helps with your studies, please kindly cite the following publication:

```
@article{wang2023dynamic,
  title={Dynamic Cognitive Diagnosis: An Educational Priors-Enhanced Deep Knowledge Tracing Perspective},
  author={Wang, Fei and Huang, Zhenya and Liu, Qi and Chen, Enhong and Yin, Yu and Ma, Jianhui and Wang, Shijin},
  journal={IEEE Transactions on Learning Technologies},
  year={2023},
  publisher={IEEE}
}
```



## Dependencies:

- python >= 3.7
- pytorch >= 1.0 (pytorch 0.4 might be OK but pytorch<0.4 is not applicable)
- numpy
- json
- sklearn



## Usage

Just `python dirt.py` or `python dneuralcdm.py`.

Please refer to the codes to see how to setup the experiments.



## Data Set Pre-process

The dataset assist2009 is provided in the folder "data/", including the original data file (skill_builder_data_corrected.csv) and the pre-processed files.

The main pre-processing includes:

1. Extract necessary columns from skill_builder_data_corrected.csv, and drop responses of which the answer_type is "open_response" or the skill_id is empty.
2. When a question contains multiple skills, the response to that question will be divided into multiple rows in the original data. Merge these rows because they correspond to the same response of a student, and collect the skill_ids from teh corresponding rows.
3.  Sort each student's responses by their timestamps and then split. The maximal length of a student's response is set to 200. When a student has more than 200 responses, split the response and regard each segment as the responses from different students.
4. Delete students with less than 15 responses.
5. Recode the IDs of students, exercieses, skills, skill_combs.
6. If a student has less than 200 responses, pad responses.
7. Divide training, validation and testing sets.



The final format of the data files is as follows:

- file_name.json = [stu1, stu2, ...]
- stu = [stu_id, log_len, [log\_1, log\_2, ...]]
- log = [order_id, exer_id, correct, [skill_id], skill_comb]



Here, log_len is the actual number of responses that the student has. The logs of each student have been padded to max_log



## Others

In the .py files, the global parameter "cross_idx" is the index of cross validation. For example, when cross_idx=1, the data files 'data/data_name/train\_1.csv', 'data/data_name/val\_1.csv' and 'data/data_name/test.csv' are used for training, validation, and testing respectively.