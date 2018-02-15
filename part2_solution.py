# Part 2 - Advanced Python
# author: Kai Chen
# date: Jan. 2017

def take(n, gen):
   list = []
   for i in range(n):
      list.append(next(gen))
   return list


def forever(gen):
   while True:
      for item in gen:
         yield item


def forever_zip(str1, str2):
   it1 = forever(list(str1))
   it2 = forever(list(str2))
   while True:
      yield (next(it1), next(it2))


for tup in take(9, forever_zip("12345", "abc")):
   print(tup)


def forever_zip_extended(*args):
   its = [forever(list(arg)) for arg in args]
   while True:
      data = [next(it) for it in its]
      yield (data)


for tup in take(15, forever_zip_extended("12345", "abc", "This is", "Python")):
   print(tup)
