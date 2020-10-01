import re, sys, os
active	= list(range(10))
born	= list(range(10, 20))


print("active",active)
print("born",born)
print("#################")

nTime			= 15
active_limit	= 12
entrance		= [g for g in born if g < nTime] 
born			= [g for g in born if not g < nTime] 

print("active",active)
print("entrance", entrance)
print("born",born)

print("#################")

nEnter			= active_limit - len(active)
active.extend(entrance[:nEnter])
born	= entrance[nEnter:]+born

print("active",active)
#print("entrance", entrance)
print("born",born)
