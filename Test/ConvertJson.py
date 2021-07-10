import os
import subprocess

d = os.listdir("F:/DataSet/Infusion/Annotations")
main_exe = "D:/Anaconda3/envs/infusion-companion/Scripts/labelme_json_to_dataset.exe"

for address in d:
    address = "F:/DataSet/Infusion/Annotations/" + address
    test = subprocess.Popen(main_exe + " " + address)
    print(test.communicate())

# print(receive)