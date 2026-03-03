import matplotlib.pyplot as plt
import numpy as np
import json
json_data = []
# ModelName='llava1.5'
# tag = '_opt4'
# ModelName='qwen'
# tag = '_opt3'
ModelName='instruct_blipv2'
tag = '_opt1'
# tag = ''
for PromptMethod in ['baseline', 'manul', 'fairPlus', ]:
# for PromptMethod in ['baseline', 'fairPlus',  'fair']:
    PromptMethod = PromptMethod+f'{tag}'
    if 'manul' in PromptMethod:
        PromptMethod = 'manul_opt1'
    with open(f'results/benchmark_scores/benchmark_results_CAPTIONING_{PromptMethod}_{ModelName}.json', "rb") as f:
        d = json.load(f)
    json_data.append(d)

labels =  ['baseline', 'fair', 'fair+']
print(json_data)
# 定义颜色列表
colors = [
    "#ECEFF1",  # 浅灰青色
    "#B0BEC5",  # 灰青色
    "#90A4AE",  # 淡蓝色
    "#607D8B"   # 深灰蓝色
]
# 准备数据
labels_ra = ['Overall Accuracy', 'RA_avg Single', 'RA_avg Two', 'RA_avg Same Gender', 'RA_avg Diff Gender']
labels_gap = ['Gender Gap Single', 'Gender Gap Two', 'Gender Gap Same Gender', 'Gender Gap Diff Gender']
ra_values = [[] for _ in range(len(json_data))]
gap_values = [[] for _ in range(len(json_data))]
for i, data in enumerate(json_data):
    ra_values[i].extend([
        data['resolution_bias']['all_images']['overall_accuracy'],
        data['resolution_bias']['single_person_images']['RA_avg'],
        data['resolution_bias']['two_person_images']['RA_avg'],
        data['resolution_bias']['two_person_images_same_gender']['RA_avg'],
        data['resolution_bias']['two_person_images_diff_gender']['RA_avg']
    ])
    gap_values[i].extend([
        data['resolution_bias']['single_person_images']['gender_gap'],
        data['resolution_bias']['two_person_images']['gender_gap'],
        data['resolution_bias']['two_person_images_same_gender']['gender_gap'],
        data['resolution_bias']['two_person_images_diff_gender']['gender_gap']
    ])

# 设置条形图的位置
x_ra = np.arange(len(labels_ra))
x_gap = np.arange(len(labels_gap))
width = 0.2  # 条形图的宽度

# 创建子图
plt.rcParams['figure.dpi'] = 300
plt.ylim(top=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# plt.ylim(0,1)

# 绘制RA_avg条形图并在顶部添加数值
for i in range(len(ra_values)):
    bars = ax1.bar(x_ra + i * width, ra_values[i], width, color=colors[i], label=labels[i])
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

# 绘制gender_gap条形图并在顶部添加数值
for i in range(len(gap_values)):
    gap_values[i] = [abs(x) for x in gap_values[i]]
    bars = ax2.bar(x_gap + i * width, gap_values[i], width, color=colors[i], label=labels[i])
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

# 添加标签和标题
# ax1.set_xlabel('Metrics')
ax1.set_ylabel('RA_avg')
# ax1.set_title('Comparison of RA_avg Across Models')
ax1.set_xticks(x_ra + width * (len(json_data) - 1) / 2)
ax1.set_xticklabels(labels_ra)
ax1.set_ylim(0,1.05)
ax1.legend()

# ax2.set_xlabel('Metrics')
ax2.set_ylabel('Gender Gap')
# ax2.set_title('Comparison of Gender Gap Across Models')
ax2.set_xticks(x_gap + width * (len(json_data) - 1) / 2)
ax2.set_xticklabels(labels_gap)
ax2.legend()
ax2.set_ylim(0,1.05)

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()
# plt.savefig(f'resimgs/{ModelName}{tag}.jpg')