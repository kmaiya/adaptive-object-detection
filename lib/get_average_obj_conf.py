import glob
from statistics import stdev

antialiased_confidences = []
normal_confidences = []
antialiased_num_detections = []
normal_num_detections = []

antialiased_txt_path = '/local/b/cam2/data/motchallenge/mot17/newaa_results/mot05/'
normal_txt_path = '/local/b/cam2/data/motchallenge/mot17/train/MOT17-05/original_detection_text_file/'

for file in list(glob.glob(antialiased_txt_path + '*.txt')):                                       
    f = open(file)
    k = f.readlines()
    antialiased_num_detections.append(len(k))

    avg = 0
    for i in k:
        i = i.rstrip('\n')
        i = i.split(',')
        avg += float(i[1])
    
    avg /= len(k)
    antialiased_confidences.append(avg)

for file in list(glob.glob(normal_txt_path + '*.txt')):                                       
    f = open(file)
    k = f.readlines()
    normal_num_detections.append(len(k))

    avg = 0
    for i in k:
        i = i.rstrip('\n')
        i = i.split(',')
        avg += float(i[1])
    
    avg /= len(k)
    normal_confidences.append(avg)

total_antialiased_detections = 0
for det in antialiased_num_detections:
    total_antialiased_detections += det

total_normal_detections = 0
for det in normal_num_detections:
    total_normal_detections += det

total_antialiased_confidence = 0
for conf in antialiased_confidences:
    total_antialiased_confidence += conf

total_normal_confidence = 0
for conf in normal_confidences:
    total_normal_confidence += conf

# print(antialiased_confidences[110:120], '\n\n')
# print(normal_confidences[110:120])
i = 100
while(i < 200):
    print('------------------------------')
    print('antialiased', stdev(antialiased_confidences[i:i+10]))
    print('normal', stdev(normal_confidences[i:i+10]))
    print('------------------------------')
    i += 10