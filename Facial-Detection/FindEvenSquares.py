import math
# helper file to get values of N for principle components that will work with restructing an image into a square

possible_num_PC = []

for i in range(500):
    # for each number, get the product of it with 500
    prod = i * 500
    # find the square root
    sqt = math.sqrt(prod)
    if sqt%1==0:
        possible_num_PC.append(i)
        print(i)

