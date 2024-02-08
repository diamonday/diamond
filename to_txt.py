import os
import cPickle
import numpy as np
import time


def main():
    start = time.time()

    filedir = "D:/Data/2021/2021-01-12/pressure point 1 odmr"
    filelist = []   # used to store root + file

    for root, dirs, files in os.walk(filedir):
        print(root)
        for file in files:
            print(dirs)
            if file.find("ODMR.pys") != -1:   # used to process ODMR data 
            #Exp: When s.find(foo) fails to find foo in s, it returns -1. Therefore, when s.find(foo) does not return -1, we know it didn't fail.
            #if file.find(".pys") != -1:
            #if file.find(".pys") != -1:
                filelist.append(os.path.join(root, file))
    print(filelist)

    for file in filelist:
        raw_file = open(file, 'rb')
        odmr = cPickle.load(raw_file)
        print(odmr.keys())
        freq = odmr['frequency']
        counts = odmr['counts']
        output = np.column_stack((freq, counts))
        processed_path = os.path.dirname(raw_file.name) + "\processed_file"
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        processed_filename = processed_path + "\\" + os.path.splitext(os.path.basename(raw_file.name))[0] + ".txt"
        np.savetxt(processed_filename, output)
    print("------%s s------" % (time.time() - start))


if __name__ == '__main__':
    main()


"""
#print(rabi['measurement']['count_data'])
rabi = cPickle.load(raw_file)
tau = rabi['measurement']['tau']
count_data = rabi['measurement']['count_data']
counts = np.sum(count_data, axis=1)
output = np.column_stack((tau, counts))"""
"""
confocal = cPickle.load(raw_file)   # confocal image counts(z)
print(confocal.keys())
output = confocal['image']
"""

"""
processed_path = os.path.dirname(raw_file.name) + "\processed_file"
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
processed_filename = processed_path + "\\" + os.path.splitext(os.path.basename(raw_file.name))[0] + ".txt"
np.savetxt(processed_filename, output)
"""