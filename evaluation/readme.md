## :bulb:Evaluation result on custom dataset: 


#1.   by running `prepare.py` result will be  `imgnamefile.txt` and `gt_labels` folder

#2.   put the `gt_labels` in `/evaluation` folder.

#3.   get detection results on val dataset.

#4.   get metrics result


### file structure should be as below:
'''
  evaluation/
    -gt_labels/
        -*.txt
    -result_classname
        -Task1_{category_name}.txt
    -batch_inference.py
    -eval.py
    -imgnamefile.txt
    -prepare.py
 '''
