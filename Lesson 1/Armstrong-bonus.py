arm = input('Armstrong Number: ')
total = 0
for i in range(int(arm)):
  total = int(arm[0])**3 + total
if total == int(arm):
  print(str(arm) + ' Is armstrong')
else:
  print(str(arm) + ' Is not armstrong')

temp = int(arm)
tot  = 0
while temp >0:
  inte = temp % 10
  tot += inte ** 3
  temp //= 10
if int(arm) == tot:
  print(str(arm) + ' Is armstrong')
else:
  print(str(arm) + ' Is not armstrong')