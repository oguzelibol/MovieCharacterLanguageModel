import os
import numpy


train = open("train.txt", 'w')
test = open("test.txt", 'w')
validate = open("valid.txt", 'w')

counter = 0
for folder in os.listdir("./dialogs"):
    if folder.startswith('.'):
        continue
    path = './dialogs/' + folder + '/'
    for file in os.listdir(path):
        if file.startswith('.'):
           continue
        with open (path + file, "r") as myfile:
            data = myfile.readlines()
            if len(data) < 20:
                continue
            counter = counter + 1
            data = data[15:]
            data = data[:-2]
            lines = []
            for t in data:
                if t.isupper():
                    continue
                if len(t) > 2 and t[0] is '(':
                    continue
                lines.append(t)
            x = numpy.random.uniform(0,100)
            if x > 25 and x < 35:
                for t in lines:
                    test.write(t)
            elif x > 67 and x < 77:
                for t in lines:
                    validate.write(t)
            else:
                for t in lines:
                    train.write(t)
print counter