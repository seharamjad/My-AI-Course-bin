# ALL PYTHON PROGRAMS 
# Organized by Sections: Strings, Lists, Dictionaries, Tuples, Loops, Math, Story

# --------------------------------------
# STRING MANIPULATION
# --------------------------------------

# 1. New string from first, middle, last character
s = input("Enter a string: ")
if len(s) == 0:
    new = ""
elif len(s) == 1:
    new = s
elif len(s) == 2:
    new = s[0] + s[1]
else:
    new = s[0] + s[len(s)//2] + s[-1]
print("Result:", new)

# 2. Count occurrences of characters
s = input("Enter string: ")
count = {}
for ch in s:
    count[ch] = count.get(ch, 0) + 1
print(count)

# 3. Reverse string
s = input("Enter string: ")
print(s[::-1])

# 4. Split on hyphens
s = input("Enter hyphen string: ")
print(s.split('-'))

# 5. Remove punctuation
import string
s = input("Enter string: ")
allowed = set(string.ascii_letters + string.digits + " ")
clean = ''.join(ch for ch in s if ch in allowed)
print(clean)


# --------------------------------------
# LIST MANIPULATION
# --------------------------------------

# 1. Reverse list
lst = [1,2,3,4,5]
print(lst[::-1])

# 2. Square items
lst = [1,2,3,4]
print([x*x for x in lst])

# 3. Remove empty strings
lst = ["a", "", "b", " ", "c"]
print([x for x in lst if x.strip() != ""])

# 4. Add item after specified
lst = ["a","b","c"]
target = "b"
lst.insert(lst.index(target)+1, "X")
print(lst)

# 5. Replace item
lst = [10,20,30]
old,new = 20,200
if old in lst:
    lst[lst.index(old)] = new
print(lst)

# 6. Largest number
lst=[5,9,2]
largest=lst[0]
for i in lst:
    if i>largest:
        largest=i
print(largest)

# 7. Second largest
lst=[10,50,20,40]
largest=max(lst)
second=None
for i in lst:
    if i!=largest:
        if second is None or i>second:
            second=i
print(second)

# 8. Largest even/odd
numbers = [12, 45, 7, 23, 56, 89, 4, 102, 33]

# 1. Find the largest Even number
# This creates a list of only evens, then finds the max
evens = [n for n in numbers if n % 2 == 0]
largest_even = max(evens) if evens else "No even numbers found"

# 2. Find the largest Odd number
# This creates a list of only odds, then finds the max
odds = [n for n in numbers if n % 2 != 0]
largest_odd = max(odds) if odds else "No odd numbers found"

print(f"Numbers: {numbers}")
print(f"Largest Even: {largest_even}")
print(f"Largest Odd: {largest_odd}")

# 9. Average
lst=[10,20,30]
t=0
for i in lst: t+=i
print(t/len(lst))

# 11. Remove duplicates
lst=[1,1,2,3,3]
unique=[]
for i in lst:
    if i not in unique: unique.append(i)
print(unique)

# 12. Odd occurring number
lst=[4,3,4,3,4]
for i in lst:
    if lst.count(i)%2!=0:
        print(i)
        break

# 13. Union of lists
lst1=[1,2,3]
lst2=[3,4,5]
union=lst1.copy()
for i in lst2:
    if i not in union: union.append(i)
print(union)

# 14. Swap first and last
lst=[10,20,30]
lst[0],lst[-1]=lst[-1],lst[0]
print(lst)

# 15. Longest word
words=["apple","banana","watermelon"]
long=max(words,key=len)
print(long)

# 16. Random numbers list
import random
n=5
lst=[random.randint(1,20) for _ in range(n)]
print(lst)


# --------------------------------------
# DICTIONARY OPERATIONS
# --------------------------------------

d={"a":1,"b":2,"c":3}
print("Key exists" if "a" in d else "Not exists")

d["d"]=4
print(d)

# Sum
s=0
for v in d.values(): s+=v
print(s)

# Product
p=1
for v in d.values(): p*=v
print(p)

# Dictionary of squares
n=5
d={}
for i in range(1,n+1): d[i]=i*i
print(d)

# Concatenate dicts
d1={"x":1,"y":2}
d2={"z":3}
for k,v in d2.items(): d1[k]=v
print(d1)


# --------------------------------------
# TUPLE OPERATIONS
# --------------------------------------

t=(1,2,3,4)
print(t[::-1])   # reverse

# Access 20
t=(10,20,30)
print(t[1])

# Swap tuples
t1=(1,2)
t2=(3,4)
t1,t2=t2,t1
print(t1,t2)

# Tuple list with squares
res=[]
for i in range(1,6): res.append((i,i*i))
print(res)

# USN tuple generation
prefix="CSU"
low,high=1,5
tuples=[]
for i in range(low,high+1): tuples.append((prefix+str(i),i))
print(tuples)


# --------------------------------------
# LOOP OPERATIONS
# --------------------------------------

# Print 10 natural numbers
i=1
while i<=10:
    print(i)
    i+=1

# Even numbers
a=int(input("Enter limit: "))
for i in range(1,a+1):
    if i%2==0: print(i)

# Odd numbers
for i in range(1,a+1):
    if i%2!=0: print(i)

# Prime numbers
num=int(input("Enter limit: "))
for i in range(2,num+1):
    prime=True
    for j in range(2,i):
        if i%j==0:
            prime=False
            break
    if prime: print(i)

# Multiplication table
n=int(input("Enter number: "))
for i in range(1,11): print(n,"x",i,"=",n*i)


# --------------------------------------
# STRING BIG ASSIGNMENT
# --------------------------------------

# Pangram
s=input("Enter string: ").lower()
alpha="abcdefghijklmnopqrstuvwxyz"
pangram=True
for ch in alpha:
    if ch not in s:
        pangram=False
        break
print("Pangram" if pangram else "Not pangram")

# Replace spaces with hyphen
s=input("Enter string: ")
out=""
for ch in s:
    out += "-" if ch==" " else ch
print(out)

# Letters in one string not both
s1=input("Enter 1st: ")
s2=input("Enter 2nd: ")
res=[]
for ch in s1:
    if ch not in s2: res.append(ch)
for ch in s2:
    if ch not in s1: res.append(ch)
print(res)

# Larger string without built-in
s1=input("String1: ")
s2=input("String2: ")
l1=l2=0
for _ in s1: l1+=1
for _ in s2: l2+=1
if l1>l2: print(s1)
elif l2>l1: print(s2)
else: print("Equal")

# Upper & Lower count
s=input("Enter string: ")
u=l=0
for ch in s:
    if ch.isupper(): u+=1
    elif ch.islower(): l+=1
print(u,l)

# Anagram
s1=input("Enter 1st: ").lower()
s2=input("Enter 2nd: ").lower()
c1={}
c2={}
for ch in s1: c1[ch]=c1.get(ch,0)+1
for ch in s2: c2[ch]=c2.get(ch,0)+1
print("Anagram" if c1==c2 else "Not anagram")

# Substring present
s=input("Main: ")
sub=input("Sub: ")
found=False
for i in range(len(s)-len(sub)+1):
    if s[i:i+len(sub)]==sub:
        found=True
        break
print("Found" if found else "Not found")

# Permutations of 3-letter string
s=input("3-letter string: ")
chars=list(s)
perms=[]
for i in range(3):
    for j in range(3):
        for k in range(3):
            if i!=j and j!=k and i!=k:
                perms.append(chars[i]+chars[j]+chars[k])
perms.sort()
print(perms)

# Length without built-in
s=input("Enter string: ")
ct=0
for _ in s: ct+=1
print(ct)

# First 2 + last 2 characters
s=input("Enter string: ")
print("" if len(s)<2 else s[:2]+s[-2:])


# --------------------------------------
# MATH OPERATIONS
# --------------------------------------

# Triangle area
import math
a=float(input("a: "))
b=float(input("b: "))
c=float(input("c: "))
s=(a+b+c)/2
print(math.sqrt(s*(s-a)*(s-b)*(s-c)))

# Quotient & Remainder
a=int(input("Dividend: "))
b=int(input("Divisor: "))
print(a)

