palin = input('Input a Palindrome: ')
if palin[::-1] == palin:
  print(str(palin) + ' Is a palindrome')
else:
  print(str(palin) + ' Is not a Palindrome')