import cv2

f = open("first_annotations.txt", 'r')
contents = f.readlines()
print(contents)
f.close()

#
path = "/data/train/"
temp = []

for i in contents:
    a = path + i
    a = a.replace(","," ")
    image_path = a.split(" ")[0]
    image = cv2.imread(image_path)
    print(image)
    width = str(image.shape[1])
    height = str(image.shape[0])
    all = a.split(" ")
    all.insert(1, height)
    all.insert(2, width)
    string = " ".join(all)
    temp.append(string)


f = open(path+"annotations_test.txt", 'w')
for i in range(len(temp)):
    f.write(temp[i])
f.close()