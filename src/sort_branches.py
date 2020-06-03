import time

start = time.time()

M_branches = []
branches = []

with open("branches.txt", "r") as file:
    for line in file:
        temp = line.strip().split("_")
        if temp[len(temp) - 1][0] == "M":
            temp_str = temp[len(temp) - 1]
            for i in range(len(temp) - 2):
                temp_str += "_" + temp[i]
            M_branches.append(temp_str)
        else:
            branches.append(line.strip())

branches_sorted = sorted(branches)
M_branches_sorted = sorted(M_branches)

with open("sorted_branches.txt", "w") as file:
    for i in range(len(branches_sorted)):
        file.write(branches_sorted[i] + "\n")
    file.write("\n\n")
    for i in range(len(M_branches_sorted)):
        file.write(M_branches_sorted[i] + "\n")

print("RUNTIME: " + str(time.time() - start) + " SECONDS.")
