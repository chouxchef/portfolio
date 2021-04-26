import matplotlib.pyplot as theplot
import random as rand
import numpy as np
np.set_printoptions(threshold=np.nan)
import copy
import operator

# part b: 8 iterations
# part c: 9 iterations
# part d: 257 iterations
# part e: 1534 iterations
rand.seed(24680)
data_size = 2000

# convert cartesian coordinates to polar
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

# convert polar coordinates to cartesian
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# generate pseudo-inverse of a matrix
def pseudo_inverse(matrix):
    pinv = np.dot(np.linalg.inv(np.dot(matrix.transpose(), matrix)), 
                  matrix.transpose())
    return pinv

def generate_line(theplot, weights, color):
    line_x_coords = np.arange(-2, 2, .1)
    line_y_coords = []
    for datapoint in line_x_coords:
        line_y_coords.append((-weights[1]/weights[2])*datapoint - (weights[0]/weights[2]))
    theplot.plot(line_x_coords, line_y_coords, c=color)
    theplot.xlim([-1.5,0.5])
    theplot.title("Problem 1, Test Set")
    theplot.xlabel("intensity")
    theplot.ylabel("symmetry")
    return theplot

def third_order_generate(theplot, weights, dataset):
    x_coords = np.arange(-1.5, 0.5, 0.01)
    y_coords = np.arange(-1.5, 0.5, 0.01)
    
    line_x_coords = []
    line_y_coords = []

    for x in x_coords:
        for y in y_coords:
            line_x_coords.append(x)
            line_y_coords.append(y)

    third_order_function_x = []
    third_order_function_y = []

    for i in xrange(len(line_x_coords)):
        threshold = third_order_eval(weights, third_order_transform([line_x_coords[i], line_y_coords[i]]))
        if abs(threshold) < .04:
            third_order_function_x.append(line_x_coords[i])
            third_order_function_y.append(line_y_coords[i])

    e_in = third_order_get_e(weights, dataset)
    print e_in
        
    theplot.scatter(third_order_function_x, third_order_function_y, c='g', 
                    marker='.')
    theplot.xlim([-1.5, 0.5])
    theplot.ylim([-1.5, 0.5])
    theplot.title("Problem 1, 3rd Order Test Data")
    theplot.xlabel("intensity")
    theplot.ylabel("symmetry")
    return theplot

def generate_data():
    x_coords = []
    y_coords = []
    sign = []
    rad = 10
    thk = 5
    sep = 5

    for x in range(0, data_size):
        # top semi-circle, filled with -1 classified points
        if np.random.randint(2) == 0:
            rho = np.random.uniform(rad, rad + thk)
            theta = np.random.uniform(0, np.pi)
            x_cart,y_cart = pol2cart(rho, theta)
            x_coords.append(x_cart)
            y_coords.append(y_cart)
            sign.append(-1)
        else:
            rho = np.random.uniform(rad, rad + thk)
            theta = np.random.uniform(np.pi, 2 * np.pi)
            x_cart,y_cart = pol2cart(rho, theta)
            x_cart += rad + .5*thk
            y_cart -= sep
            x_coords.append(x_cart)
            y_coords.append(y_cart)
            sign.append(1)

    dataset = []
    for i in xrange(data_size):
        dataset.append([])
        dataset[i].append(1) # bias term
        dataset[i].append(x_coords[i])
        dataset[i].append(y_coords[i])

    return (dataset, sign)

def determine_sign(weights, x, y):
    product = weights[0] + weights[1] * x + weights[2] * y
    return np.sign(product)

def find_misclassified(weights, dataset):
    #print weights
    #print (dataset[0])
    for index,point in enumerate(dataset):
        if (determine_sign(weights, dataset[index][0], dataset[index][1]) != dataset[index][2]):
            return index

    return None

def third_order_determine(weights, data_point):
    product = third_order_eval(weights, data_point)
    return np.sign(product)
        

def third_order_find(weights, dataset):
    for index,point in enumerate(dataset):
        if (third_order_determine(weights, point) != dataset[index][10]):
            return index

    return None

def perceptron(learning_weights, dataset):
    incorrect_point = find_misclassified(learning_weights, dataset)
    print("first point chosen", incorrect_point)
    iteration = 0
    while (incorrect_point is not None):
        point_copy = copy.deepcopy(dataset[incorrect_point])
        point_copy.pop(2)
        point_copy.insert(0, 1)

        # multiply by Yt)
        for i in range(len(point_copy)):
            point_copy[i] = dataset[incorrect_point][2] * point_copy[i]
        print("Y(t) * X(t)",point_copy)
            
        for i in range(len(point_copy)):
            learning_weights[i] += point_copy[i]

        incorrect_point = find_misclassified(learning_weights, dataset)
        print("next wrong point", incorrect_point)
        iteration += 1

    return iteration

def new_find_misclassified(weights, dataset):
    misclassified_count = 0
    for index,point in enumerate(dataset):
        if (determine_sign(weights, dataset[index][0], dataset[index][1]) != dataset[index][2]):
            misclassified_count += 1

    return misclassified_count

def third_order_find_new(weights, dataset):
    misclassified_count = 0
    for index,point in enumerate(dataset):
        if (third_order_determine(weights, point) != dataset[index][10]):
            misclassified_count += 1

    return misclassified_count

def pocket(learning_weights, dataset):

    best_weights = list(learning_weights)
    incorrect_point = find_misclassified(learning_weights, dataset)
    #print("first point chosen", incorrect_point)
    iteration = 0
    while (incorrect_point is not None and iteration < 1000):
        point_copy = copy.deepcopy(dataset[incorrect_point])
        np.delete(point_copy, 2)
        np.insert(point_copy, 0, 1)

        #print point_copy

        # multiply by Yt)
        for i in range(len(point_copy)):
            point_copy[i] = dataset[incorrect_point][2] * point_copy[i]
        #print("Y(t) * X(t)",point_copy)
            
        #print learning_weights

        for i in range(len(point_copy)):
            learning_weights[i] += point_copy[i]

        #print len(learning_weights)
        #print len(dataset[0])
        #print best_weights

        # Evaluate E_in(learning_weights) (find ALL misclassified points)
        # If E_in(learning_weights) < E_in(best_weights)
        # then best_weights = learning_weights

        #print get_e_in(learning_weights, dataset)
        #print get_e_in(best_weights, dataset)

        if get_e_in(learning_weights, dataset) < get_e_in(best_weights, dataset):
            best_weights = list(learning_weights)


        # prepare for next iteration
        incorrect_point = find_misclassified(learning_weights, dataset)
        #print("next wrong point", incorrect_point)
        iteration += 1

    print get_e_in(best_weights, dataset)
    return best_weights

def third_order_transform(dataset):
    third_order_dataset = []
    print len(dataset)
    if len(dataset) > 10:
        for index,point in enumerate(dataset):
            third_order_dataset.append([1, dataset[index][0], dataset[index][1], 
                                        dataset[index][0]**2, 
                                        dataset[index][0]*dataset[index][1], 
                                        dataset[index][1]**2, dataset[index][0]**3,
                                        (dataset[index][0]**2)*dataset[index][1],
                                        dataset[index][0]*(dataset[index][1]**2), 
                                        dataset[index][1]**3])

    else:
        third_order_dataset.append(1)
        third_order_dataset.append(dataset[0])
        third_order_dataset.append(dataset[1])
        third_order_dataset.append(dataset[0]**2)
        third_order_dataset.append(dataset[0]*dataset[1])
        third_order_dataset.append(dataset[1]**2)
        third_order_dataset.append(dataset[0]**3)
        third_order_dataset.append((dataset[0]**2)*dataset[1])
        third_order_dataset.append(dataset[0]*(dataset[1]**2))
        third_order_dataset.append(dataset[1]**3)

    return third_order_dataset
    

def third_order_pocket(learning_weights, dataset):

    best_weights = list(learning_weights)
    incorrect_point = third_order_find(learning_weights, dataset)
    #print("first point chosen", incorrect_point)
    iteration = 0
    while (incorrect_point is not None and iteration < 1000):
        point_copy = copy.deepcopy(dataset[incorrect_point])
        print point_copy
        np.delete(point_copy, 10)
        print point_copy
        #np.insert(point_copy, 0, 1)

        #print point_copy

        # multiply by Yt)
        for i in range(len(point_copy)):
            point_copy[i] = dataset[incorrect_point][10] * point_copy[i]
        #print("Y(t) * X(t)",point_copy)
            
        #print learning_weights

        print point_copy.shape

        for i in range(len(point_copy)):
            learning_weights[i] += point_copy[i]

        #print learning_weights
        #print best_weights

        # Evaluate E_in(learning_weights) (find ALL misclassified points)
        # If E_in(learning_weights) < E_in(best_weights)
        # then best_weights = learning_weights

        #print get_e_in(learning_weights, dataset)
        #print get_e_in(best_weights, dataset)

        if third_order_get_e(learning_weights, dataset) < third_order_get_e(best_weights, dataset):
            best_weights = list(learning_weights)


        # prepare for next iteration
        incorrect_point = third_order_find(learning_weights, dataset)
        #print("next wrong point", incorrect_point)
        iteration += 1

    #print get_e_in(best_weights, dataset)
    return best_weights



def linear_regression(dataset, target_vector):
    data_matrix = np.array(dataset)
    #print data_matrix.shape
    augmented_data = np.insert(data_matrix, 0, 1, axis=1)
    
    # compute pseudoinverse of data matrix
    #print augmented_data
    pseudo = pseudo_inverse(augmented_data)
    #print pseudo.shape
    wlin = np.dot(pseudo, np.array(target_vector))
    #print wlin
    return wlin

def third_order_regression(dataset, target_vector):
    data_matrix = np.array(dataset)
    #augmented_data = np.insert(data_matrix, 0, 1, axis=1)
    # compute pseudoinverse of data matrix
    pseudo = pseudo_inverse(data_matrix)
    wlin = np.dot(pseudo, target_vector)
    return wlin    

def third_order_eval(weights, data_point):

    #print data_point.shape
    #print len(weights)

    dot_product = np.dot(weights, data_point[0:10])

    return dot_product

def process_digits():
    data_file = open("ZipDigits.test")
    data_matrix = []
    correct_digits = []

    for line in data_file:
        temp = []
        if line[0] == '1' or line[0] == '5':
            if line[0] == '1':
                correct_digits.append(1)
            else:
                correct_digits.append(-1)
            one_line = line.strip()
            for word in one_line[7::].split(" "):
                temp.append(word)
            data_matrix.append(temp)

    data_file.close()

    for inner in data_matrix:
        for index, string in enumerate(inner):
            inner[index] = float(string)

    return data_matrix,correct_digits

def symmetry_matrix(data_matrix):
    mirror = [sublist[::-1] for sublist in data_matrix]

    return mirror

def get_intensity(data_matrix):
    average_intensity = []
    for digit_data in data_matrix:
        digit_average = sum(digit_data) / float(len(digit_data))
        average_intensity.append(digit_average)

    return average_intensity

def get_symmetry(data_matrix, mirror_data_matrix):
    difference_matrix = np.array(data_matrix) - np.array(mirror_data_matrix)
    absolute_matrix = np.absolute(difference_matrix)
    average_absolute_matrix = np.mean(absolute_matrix, axis=1)
    
    return np.negative(average_absolute_matrix)
   
def get_e_in(weights, dataset):
    incorrect_fraction = new_find_misclassified(weights, dataset)
    e_in = incorrect_fraction / float(len(dataset))

    return e_in
        
def third_order_get_e(weights, dataset):
    incorrect_fraction = third_order_find_new(weights, dataset)
    e_in = incorrect_fraction / float(len(dataset))

    return e_in


def main():

    #target_weights = [1.0, 3.0, -5.0]
    #learning_weights = [0.0, 0.0, 0.0]
    learning_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    

    #dataset,target_vector = generate_data()
    dataset,correct_digits = process_digits()
    mirror_dataset = symmetry_matrix(dataset)

    average_intensity = get_intensity(dataset)
    average_symmetry = get_symmetry(dataset, mirror_dataset)

    feature_dataset = np.column_stack((average_intensity, average_symmetry))
    third_order_dataset = third_order_transform(feature_dataset)

    print third_order_dataset[0]
    data_matrix =  np.column_stack((feature_dataset, correct_digits))
    third_order_data_matrix = np.column_stack((third_order_dataset, 
                                               correct_digits))

    wlin = linear_regression(feature_dataset, correct_digits)
    third_order_wlin = third_order_regression(third_order_dataset, correct_digits)
    #print third_order_data_matrix.shape

    best_weights = pocket(wlin, data_matrix)
    #third_order_best_weights = third_order_pocket(third_order_wlin, 
    #                                              third_order_data_matrix)

    


    #print len(dataset)

    #print average_symmetry

    #print correct_digits
    #print average_symmetry.shape



    #count = perceptron(learning_weights, dataset)
    #print count

    #theplot.scatter([row[1] for row in third_order_function_points],
    #                [row[2] for row in third_order_function_points], c='g', marker='o')

    # make a scatter plot
    for i in xrange(len(average_intensity)):
        if data_matrix[i][2] > 0:
            theplot.scatter(data_matrix[i][0], data_matrix[i][1], c='b', marker='o')
        else:
            theplot.scatter(data_matrix[i][0], data_matrix[i][1], c='r', marker='x')
    # draw the target line f
    #plt = generate_line(theplot, best_weights, 'g')
    #plt = generate_line(theplot, wlin, 'g')
 
    plt = third_order_generate(theplot, third_order_wlin, third_order_data_matrix)


    #theplot.show()
    plt.show()

if __name__ == '__main__':
    main()
