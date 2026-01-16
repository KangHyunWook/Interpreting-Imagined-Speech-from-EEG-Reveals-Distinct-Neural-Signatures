import numpy as np
import matplotlib.pyplot as plt

loss_o_path = '/home/Hyunwook/COPA-main/seed_acc_all_subjects_loss_o.csv'
loss_x_path = '/home/Hyunwook/COPA-main/seed_acc_all_subjects_loss_x.csv'


loss_o_accs=[]




import numpy as np
import matplotlib.pyplot as plt

# Example data
np.random.seed(0)
n_sub = 21
subjects = [f"Sub 0{i+1}" if i < 9 else f"Sub {i+1}" for i in range(n_sub)]

caseA=[90.01, 75.07, 72.51, 72.08, 71.78, 64.33, 62.63, 61.27, 62.41, 64.47, 64.87, 64.81, 66.77, 66.81, 64.26, 64.87, 66.86, 64.79, 63.17, 67.08, 65.73]
caseB=[88.96, 80.00, 80.70, 80.18, 91.45, 50.59, 50.01, 76.59, 82.28, 82.50, 75.86, 77.18, 84.97, 77.92, 91.55, 80.42, 79.21, 92.25, 73.39, 82.57, 79.15]
# Compute group averages and SDs
meanA, stdA = np.mean(caseA), np.std(caseA)
meanB, stdB = np.mean(caseB), np.std(caseB)

# Add "Average" entry
subjects.append("AVG")
caseA = np.append(caseA, meanA)
caseB = np.append(caseB, meanB)
stdsA = np.append(np.zeros(n_sub), stdA)
stdsB = np.append(np.zeros(n_sub), stdB)

# Plot setup
x = np.arange(len(subjects))
width = 0.35

plt.figure(figsize=(11, 5))
bars1 = plt.bar(x - width/2, caseA, width, yerr=stdsA, capsize=4,
                label="SVM", color="#d62728", edgecolor="k")
bars2 = plt.bar(x + width/2, caseB, width, yerr=stdsB, capsize=4,
                label="Transformer", color="#1f77b4", edgecolor="k")

# Highlight the average bars visually (optional)
bars1[-1].set_color("#d62728")  # darker blue
bars2[-1].set_color("#1f77b4")  # darker orange

plt.ylabel("Accuracy (%)")

plt.xticks(x, subjects, rotation=45, ha="right")
plt.legend(loc='upper right',ncol=2, bbox_to_anchor=(1,1.1))
plt.tight_layout()
plt.savefig('reg_abl.png', bbox_inches='tight', dpi=300)
plt.show()