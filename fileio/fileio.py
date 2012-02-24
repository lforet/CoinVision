import csv
import sys



test_list = [1,3,3,5,5,6,7,89,9]
print test_list
print sorted(test_list)
print test_list.sort()

"""
f = open('testdata.txt', 'rt')
try:
	reader = csv.reader(f)
	for row in reader:
		print row
		print reader.line_num
	print row[0]
finally:
    f.close()
"""
