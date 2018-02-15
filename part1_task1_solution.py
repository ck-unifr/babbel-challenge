# Part 1- Basic Python
# author: Kai Chen
# date: Jan. 2017

# Task 1

table = [
         {'age': 32, 'gender':'m', 'loc':'Germany', 'val':4233},
         {'age': 23, 'gender': 'f', 'loc': 'US', 'val': 11223},
         {'age': 31, 'gender': 'f', 'loc': 'France', 'val': 3234},
         {'age': 41, 'gender': 'm', 'loc': 'France', 'val': 2230},
         {'age': 19, 'gender': 'm', 'loc': 'Germany', 'val': 42},
         {'age': 21, 'gender': 'f', 'loc': 'France', 'val': 3315},
         {'age': 23, 'gender': 'm', 'loc': 'Italy', 'val': 520},
         ]


def group_aggregate(groupby, field, agg, table):
   dict = {key: [] for key in (item.get(groupby) for item in table)}

   for item in table:
      key = item.get(groupby)
      dict[key].append(item.get(field))

   dict = {key: agg(value) for key, value in dict.items()}

   print('aggregating %s (%s)' % (field, agg.__name__))
   for key, item in dict.items():
      print(key, ':', item)

   return


group_aggregate(groupby = 'gender', field = 'age', agg = sum, table = table)
print()
group_aggregate(groupby = 'loc', field = 'age', agg = min, table = table)