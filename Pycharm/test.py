n=3
for i in range(n):
    print(i)
print('cut')
while n<9:
    print(n)
    n+=1
if n==9:
    print('mbut')

numberList = [1, 2, 3]
strList = ['one', 'two', 'three']

# No iterables are passed
result = zip()

# Converting itertor to list
resultList = list(result)
print(resultList)

# Two iterables are passed
result = zip(numberList, strList)

# Converting itertor to set
resultSet = set(result)
print(resultSet)