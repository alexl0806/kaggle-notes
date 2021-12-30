# functions and getting help

# help function returns the description of a functions
help(round)

# can use key to apply a function to each term
def mod_5(x):
    return x%5

max(100, 51, 14, key=mod_5)

# here we have num_friends, which is an optional argument, defaulted to 3
def to_smash (total_candies, num_friends = 3):
    return total_candies % num_friends

# we can use the type function to determine the type of a variable
type(39)

# booleans and conditionals

# and is evaluated before or

# type conversions
int('3') # gives 3
float(4) # gives 4.0
bool("asf") # gives true, only string that gives false is ""

# shorthand way of inputting an if-statement inside a print-statement
print("Splitting", total_candies, "candy" if total_candies == 1 else "candies")

# these two return statements are equivalent, the int conversions of the booleans are implicitly done
return (int(ketchup) + int(mustard) + int(onion)) == 1
return (ketchup + mustard + onion)

# Lists

# sorted returns a sorted version of a list
sorted(planets)

# modifies a list by adding an item to the end
planets.append('Pluto')

# removes and returns the last element of the List
planets.pop()

# can search for the index of an element
planets.index('Earth')

# checking to see if a list contains a particular value
"Earth" in planets # => Bool

# tuples are created using parentheses, and are immutable (cannot be modified)
t = (1,2,3)

# tuples are often used for functions that have multiple return values
x = 0.125
x.as_integer_ratio() # this will output the tuple (1, 8)

numerator, denominator = x.as_integer_ratio() # individual assignment

# using tuples to swap two variables
a = 1
b = 0
a, b = b, a
print (a, b)

# loops and list comprehensions

# in the for-loop, we specify the variable name (planet) and the set of values to loop over (planets)
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for planet in planets: # anything to the right of "in" just has to be an object that supports iteration
    print(planet, end=' ') # print all on same line

# range() function returns a sequence of numbers
for i in range(5):
    print(i)

# while-loops iterate until some condition is met
i=0

while i < 10:
    print(i, end=' ')
    i += 1

squares = [n**2 for n in range(10)] # example of a list-comprehension
short_planets = [planet for planet in planets if len(planet) < 6] # adding an if statement
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6] # making it more complex

# example of a function using list comprehensions
def count_negatives(nums):
    return len([num for num in nums if num < 0])

# another list comprehension example
def has_lucky_number(nums):
    return any([num % 7 == 0 for num in nums])

# strings and dictionaries

# tricks with quotations
'Pluto\'s a planet!' # print apostrophe with single quotes
"That's \"cool\"" # That's "cool"
"Look, a mountain: /\\" # Look, a mountain: /\
"1\n2 3" # 1 newline 2 3"

str.split() # splits a string into a list of smaller things, breaking on whitespace by default
str.join() # joins a string together

position = 9
"{}, you'll always be the {}th planet to me.".format(planet, position) # replaces the brackets with values

pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390

#         2 decimal points   3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population)

# example of a dictionary
numbers = {'one':1, 'two':2, 'three':3}

# adding initials + creating dictionary
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
planet_to_initial

# to get the keys and the values
dict.keys()
dict.values()
dict.items() # iterate over keys and values simultaneously

# function that can search for a keyword in a list of strings
def word_search(doc_list, keyword):
    lst = []

    for i, doc in enumerate(doc_list):
        tokens = doc.split()
        normalized = [token.rstrip('.,').lower() for token in tokens]

        if keyword.lower() in normalized:
            lst.append(i)

    return lst

# function that can search for multiple keywords in a List
def multi_word_search(doc_list, keywords):
    return {keyword: word_search(doc_list, keyword) for keyword in keywords}

# working with external libraries

import math # example of importing a library
import math as mt # importing as a shorter alias
from math import * # allows us to just use pi and stuff without the math

type() # tells us what something is
dir() # tells us what we can do with it
help() # tells us more about something
