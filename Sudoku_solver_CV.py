import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
from random import shuffle
from math import floor


# Function definitions 
# I dont want the model to learn with each fit so it will be a function
def CNN_builder():
    inputs= keras.Input(shape=(50,50,1), name='Input layer')
    x = keras.layers.Conv2D(filters= 32, kernel_size = 3, strides = (1,1) , padding='valid',  activation='relu' ,name="conv_layer_1")(inputs)
    x = keras.layers.MaxPool2D(pool_size=2, name="pooling_1")(x)
    x = keras.layers.Conv2D(filters= 64, kernel_size = 3, activation='relu', name="conv_layer_2")(x)
    x = keras.layers.MaxPool2D(pool_size=2, name="pooling_2")(x)
    x = keras.layers.Conv2D(filters= 128, kernel_size = 3, activation='relu', name="conv_layer_3")(x)
    x = keras.layers.Flatten(name="flattening_layer")(x)
    x = keras.layers.Dense(units= 64, activation='relu')(x)
    outputs = keras.layers.Dense(units= 10, activation='softmax', name='output_layer')(x)
    model = keras.Model(inputs= inputs , outputs=outputs, name='SCNN')
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
    return model

# next two functions use backtracking to solve the board once the computer understands the picture
def is_valid(grid, r, c, k):
    not_in_row = k not in grid[r]
    not_in_column = k not in [grid[i][c] for i in range(9)]
    not_in_box = k not in [grid[i][j] for i in range(r//3*3, r//3*3+3) for j in range(c//3*3, c//3*3+3)]
    return not_in_row and not_in_column and not_in_box

def solve(grid, r=0, c=0):
    if r == 9:
        return True
    elif c == 9:
        return solve(grid, r+1, 0)
    elif grid[r][c] != 0:
        return solve(grid, r, c+1)
    else:
        for k in range(1, 10):
            if is_valid(grid, r, c, k):
                grid[r][c] = k
                if solve(grid, r, c+1):
                    return True
                grid[r][c] = 0
        return False


# -------------------------------------------------------------------------------------------- 
#  I tried to use the mnist dataset initially, but the computer just wouldnt get a handful of numbers righ so im making my own dataset 
#  I dont not care if the model overfits to oblivion and back -> it just needs to be right.

transformed = []
output = []
for i in range(1,10):

    img_path = os.getcwd() +"\\RawNumbers\\"+str(i)+".png"
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    for y in range(5):
        y += 3
        for x in range(5):
            x += 3
            transformed.append(img[y:y+50,x:x+50])
            output.append(i)

zipped = list(zip(transformed, output))

# The im going to train on 100k and validate on 12.5k images
zipped2 = zipped * 50
shuffle(zipped2)

trainx, testx, trainy, testy, = list(), list(), list(), list()

# populate the new arrays
for x, y in zipped2[:floor(len(zipped2) * 0.9)]:
    trainx.append(x)
    trainy.append(y)

for x, y in zipped2[floor(len(zipped2) * 0.9):]:
    testx.append(x)
    testy.append(y)

# convert the data into numpy arrays 
trainx, testx, trainy, testy, = np.asarray(trainx), np.asarray(testx), np.asarray(trainy), np.asarray(testy)

# Normalize the Datasets
trainx = trainx/255
testx = testx/255

# reshpe the data to 100k, 50, 50, 1 & 12.5k, 50, 50, 1
trainx = trainx.reshape((floor(len(zipped2) * 0.9), 50, 50, 1))
testx = testx.reshape((floor(len(zipped2) * 0.1), 50, 50, 1))

# Create the one hot code encoding for the labels
trainy= tf.keras.utils.to_categorical(trainy, 10)
testy= tf.keras.utils.to_categorical(testy, 10)

# Start building the model
model = CNN_builder()
history = model.fit(x=trainx, y=trainy,
          validation_data=(testx, testy),
          epochs=75, batch_size=64)

# load the image 
img_path = os.getcwd() +"\\SudokuBoards\\M_board.png"                       # change the name of the board right here --------------------------------
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# The board is not even so the coordinates need to be hardcoded
X_cords = [(22, 143),(146, 269),(272, 393),(400, 520),(524, 646),(650, 770),(777, 898),(901, 1024),(1027, 1148)]
Y_cords = [(458, 579),(582, 705),(708, 829),(838, 956),(960, 1082),(1086, 1206),(1213, 1334),(1337, 1460),(1463, 1584)]

matrix = [[],[],[],[],[],[],[],[],[]]

# Traverse by slicing x[j][s&e] y[i][s&e]
for i in range(9):
    for j in range(9):
        sub_img = img[Y_cords[i][0]:Y_cords[i][1], X_cords[j][0]:X_cords[j][1]]

        # convert the subimage to a 50x50 picture
        lil_img = cv.resize(sub_img, (50, 50))

        if np.all(lil_img == lil_img[0][0]):
            matrix[i].append(0)
        else:
            lil_img = lil_img/255
            image = np.expand_dims(lil_img, axis=0)
            prediction = model.predict(image)
            matrix[i].append(np.argmax(prediction))

print("This is the board that has been interpreted by the computer:")
print(*matrix, sep='\n')
print()
solve(matrix)
print("Solved board:")
print(*matrix, sep='\n')