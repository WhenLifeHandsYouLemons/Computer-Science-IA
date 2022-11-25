import os

PATH = "datasets/clarencezhao/eval/zero/"

files = os.listdir(f"{PATH}")

i = 958
done = False
for file in files:
    while done is not True:
        try:
            os.rename(f"{PATH}{file}", f"{PATH}{i}.jpg")
            done = True
        except FileExistsError:
            print("Skipped number:", i)
            done = False
        i += 1
    done = False

print(i)