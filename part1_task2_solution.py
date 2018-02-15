# Part 1- Basic Python
# author: Kai Chen
# date: Jan. 2017

# Task 2

table = [
         {'age': 32, 'gender':'m', 'loc':'Germany', 'val':4233},
         {'age': 23, 'gender': 'f', 'loc': 'US', 'val': 11223},
         {'age': 31, 'gender': 'f', 'loc': 'France', 'val': 3234},
         {'age': 41, 'gender': 'm', 'loc': 'France', 'val': 2230},
         {'age': 19, 'gender': 'm', 'loc': 'Germany', 'val': 42},
         {'age': 21, 'gender': 'f', 'loc': 'France', 'val': 3315},
         {'age': 23, 'gender': 'm', 'loc': 'Italy', 'val': 520},
         ]


def pretty_print(table):
   if(len(table) == 0):
      return

   # for key in table[0].keys():
   #   print(key)
   """ Note:
   If we uncomment the above lines, we can see that every time we run the script, the order of the keys is changed.
   Because each pair of key and value has a hash value, this hash value is generated each time.
   Therefore, the orders of the hash values (also the order of key, value pair) are different."""


   # Sort columns by the order of alphabetic.
   key_list = [key for key, _ in table[0].items()]
   key_list.sort()

   # Put the table into a matrix. The keys are the first row of that matrix. The rest rows are the content of the table.
   content = []
   content.append([key for key in key_list])
   for item in table:
      content.append([str(item.get(key)) for key in key_list])

   # Compute the maximum column width of the matrix.
   col_width = [max(len(x) for x in col) for col in zip(*content)]


   # Print the table
   print("+" + "+".join("{:{}}".format('-'*col_width[i], col_width[i])
                           for i, x in enumerate(content[0])) + "+")
   print("|" + "|".join("{:>{}}".format(x, col_width[i])
                        for i, x in enumerate(content[0])) + "|")
   print("+" + "+".join("{:{}}".format('-' * col_width[i], col_width[i])
                        for i, x in enumerate(content[0])) + "+")

   for line in content[1:]:
      print("|" + "|".join("{:>{}}".format(x, col_width[i])
                              for i, x in enumerate(line)) + "|")

   print("+" + "+".join("{:{}}".format('-' * col_width[i], col_width[i])
                        for i, x in enumerate(content[0])) + "+")
   return


pretty_print(table)


"""
Question: why the order of the columns in the table changes every time ?
Answer:
In python, in order to create a dictionary, a hash function is needed which takes the information in a key object and uses it to produce an integer, called a hash value.
This hash value is then used to determine which (key, value) pair should be placed into.
Each time, we create a new dictionary, new hash values are generated for the keys.
Therefore, the order of keys changes every time when we run the script.
"""