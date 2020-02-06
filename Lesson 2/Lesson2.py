from typing import List, Dict
# Problem 1 = convert list of student weight in pounds to kilograms
sPounds =[]
sKilograms = []
students = int(input("How many students? : "))
for i in range(students):
  weight = int(input("Please enter student weight in lbs: "))
  sPounds.append(weight)  
for i in range(len(sPounds)):
  tmp = sPounds[i]
  sKilograms.append(tmp/2.2046)
for i in range(len(sKilograms)):
  print("Student " + str(int(i)) + " Weight in Kilogram: " + str(int(sKilograms[i])))

# Problem 2 = print every other character of input string
inputStr = input("Please enter a string :")
def string_alternative(String):
  strBldr = ""
  altStr = []
  for i in range(0, len(String), 2):
    altStr.append(String[i])
  for i in range(len(altStr)):
    strBldr = str(strBldr + altStr[i])
  print(strBldr)

string_alternative(inputStr)

# Problem 3

def find_word_count(file_name: str) -> Dict[str, int]:
    """
    Get word_count in a file for each line
    :param file_name:
    :return: Dict of str:int representing word count for the file
    """
    try:
        word_count_dict = dict()
        fh = open(file=file_name, mode='r')
        for line in fh:
            for word in line.split(sep=" "):
                word = word.strip()
                word_count_dict[word] = word_count_dict.get(word, 0) + 1
        return word_count_dict
    except FileNotFoundError:
        print("{} could not be found".format(file_name))

def append_word_counts_to_file(file_name: str, word_count_dict: Dict[str, int]):
    """
    Append word counts to the bottom of the file
    :param file_name:
    :param word_count_dict:
    :return: None
    """
    try:
        fh = open(file=file_name, mode='a')
        fh.write("\n\n" + ("=" * 10) + "\n" + "WORD COUNTS:\n")
        for word, count in word_count_dict.items():
            fh.write(word + " : " + str(count) + "\n")
    except FileNotFoundError:
        print("{} could not be found".format(file_name))

def run_word_count_program(file_name: str):
    append_word_counts_to_file(file_name, find_word_count(file_name))
if __name__ == "__main__":
    run_word_count_program("string.txt")