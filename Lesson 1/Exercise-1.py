string1 = input('Enter a string: ')
# Take the input and find a h or o and replace it with empty string.
sliced1 = string1.replace('h','')
sliced2 = sliced1.replace('o','')
# reverse the string through a join command
print(''.join(reversed(sliced2)))
# reverse through an array operation
print(sliced2[::-1])
int1 = input('Integer1: ')
int2 = input('Integer2: ')
# Take the 2 inputs and sum them, then multiply
print('Sum: ' + str(int(int1) + int(int2)) + '\r\n'+ 
'Multiply: ' + str(int(int1) * int(int2)))
string2 = input('Python string input: ')
# Find all instances of 'python' and replace it with 'pythons'
print(string2.replace('python','pythons'))

