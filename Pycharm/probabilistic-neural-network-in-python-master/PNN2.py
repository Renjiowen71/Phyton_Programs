import numpy as np
import pandas as pd

tabletrain = pd.read_csv("training_data.csv", delimiter = ';' )
table1 = pd.read_csv("zoo_data.csv", delimiter = ';')
tabletrain_attribute = tabletrain.drop(tabletrain.columns[[0,17]],axis = 1)
table1_attribute = table1.drop(table1.columns[[0,17]],axis = 1)
arraytrain_att = tabletrain_attribute.values
array1_att = table1_attribute.values
vector_length = arraytrain_att.shape[1]
cleaned_data = arraytrain_att

groups = tabletrain.groupby('type')
number_of_classes = len(groups)
dictionary_of_sum = {}
number_of_features  = vector_length
sigma = 1
increament_current_row_in_matrix = 0
temp = 0
temp_sum = 0

for x in array1_att:
    point_want_to_classify = x
    increament_current_row_in_matrix = 0
    for k in range(1, number_of_classes + 1):

        # 4.1 Initiate the sume to zero
        dictionary_of_sum[k] = 0
        number_of_data_point_from_class_k = len(groups.get_group(k))

        # ** PATTERN LAYER OF PNN **
        # 5. Loop via the number of training example in class i
        # 5.1 - Declare a temporary variable to hold the sum of gaussian distribution sum
        temp_summnation = 0.0

        # 6. Loop via number of points in the class - NUMBER OF POINTS IN THE CLASS!
        for i in range(1, number_of_data_point_from_class_k + 1):

            # 6.1 - Implementation of getting Gaussians
            for j in range(0, vector_length):
                temp = (point_want_to_classify[j] - cleaned_data[increament_current_row_in_matrix][j]) * (point_want_to_classify[j] - cleaned_data[increament_current_row_in_matrix][j])

                temp_sum = temp_sum + temp
            temp_sum = -1 * temp_sum / (2 * np.power(sigma, 2))

            # 6.2 - Implementation of Sum of Gaussians
            temp_summnation = temp_summnation + temp_sum

            # 6.3 - Increamenting the row of the matrix to get the next data point
            increament_current_row_in_matrix = increament_current_row_in_matrix + 1

        # 7. Finally - For K class - the Probability of current data point belonging to that class
        dictionary_of_sum[k] = temp_summnation
    # 8. Get the classified class
    classified_class = str(max(dictionary_of_sum, key=dictionary_of_sum.get))
    print(classified_class)