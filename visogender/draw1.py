import matplotlib.pyplot as plt
import numpy as np
def data_deal():
    data = {'llava1.5': [{'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'llava1.5'},
                          'resolution_bias': {'all_images': {'overall_accuracy': 0.78},
                                              'single_person_images': {'RA_avg': 0.92, 'gender_gap': 0.15},
                                              'two_person_images': {'RA_avg': 0.64, 'gender_gap': 0.73},
                                              'two_person_images_same_gender': {'RA_avg': 0.73, 'gender_gap': 0.54},
                                              'two_person_images_diff_gender': {'RA_avg': 0.54, 'gender_gap': 0.93}}},
                         {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'llava1.5'},
                          'resolution_bias': {'all_images': {'overall_accuracy': 0.8},
                                              'single_person_images': {'RA_avg': 0.93, 'gender_gap': 0.14},
                                              'two_person_images': {'RA_avg': 0.66, 'gender_gap': 0.67},
                                              'two_person_images_same_gender': {'RA_avg': 0.8, 'gender_gap': 0.41},
                                              'two_person_images_diff_gender': {'RA_avg': 0.54, 'gender_gap': 0.93}}},
                         {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'llava1.5'},
                          'resolution_bias': {'all_images': {'overall_accuracy': 0.82},
                                              'single_person_images': {'RA_avg': 0.96, 'gender_gap': 0.09},
                                              'two_person_images': {'RA_avg': 0.68, 'gender_gap': 0.65},
                                              'two_person_images_same_gender': {'RA_avg': 0.8, 'gender_gap': 0.39},
                                              'two_person_images_diff_gender': {'RA_avg': 0.55, 'gender_gap': 0.91}}}],
            'qwen': [{'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'qwen'},
                      'resolution_bias': {'all_images': {'overall_accuracy': 0.4},
                                          'single_person_images': {'RA_avg': 0.39, 'gender_gap': 0.59},
                                          'two_person_images': {'RA_avg': 0.41, 'gender_gap': 0.3},
                                          'two_person_images_same_gender': {'RA_avg': 0.62, 'gender_gap': 0.53},
                                          'two_person_images_diff_gender': {'RA_avg': 0.18, 'gender_gap': 0.03}}},
                     {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'qwen'},
                      'resolution_bias': {'all_images': {'overall_accuracy': 0.68},
                                          'single_person_images': {'RA_avg': 0.7, 'gender_gap': 0.27},
                                          'two_person_images': {'RA_avg': 0.66, 'gender_gap': 0.05},
                                          'two_person_images_same_gender': {'RA_avg': 0.9, 'gender_gap': 0.09},
                                          'two_person_images_diff_gender': {'RA_avg': 0.4, 'gender_gap': -0.02}}},
                     {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'qwen'},
                      'resolution_bias': {'all_images': {'overall_accuracy': 0.82},
                                          'single_person_images': {'RA_avg': 0.9, 'gender_gap': 0.1},
                                          'two_person_images': {'RA_avg': 0.74, 'gender_gap': 0.05},
                                          'two_person_images_same_gender': {'RA_avg': 0.94, 'gender_gap': 0.05},
                                          'two_person_images_diff_gender': {'RA_avg': 0.54, 'gender_gap': 0.02}}}],
            'instruct_blipv2': [{'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'instruct_blipv2'},
                                 'resolution_bias': {'all_images': {'overall_accuracy': 0.72},
                                                     'single_person_images': {'RA_avg': 0.82, 'gender_gap': 0.35},
                                                     'two_person_images': {'RA_avg': 0.63, 'gender_gap': 0.72},
                                                     'two_person_images_same_gender': {'RA_avg': 0.74,
                                                                                       'gender_gap': 0.51},
                                                     'two_person_images_diff_gender': {'RA_avg': 0.51,
                                                                                       'gender_gap': 0.94}}},
                                {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'instruct_blipv2'},
                                 'resolution_bias': {'all_images': {'overall_accuracy': 0.76},
                                                     'single_person_images': {'RA_avg': 0.83, 'gender_gap': 0.24},
                                                     'two_person_images': {'RA_avg': 0.7, 'gender_gap': 0.34},
                                                     'two_person_images_same_gender': {'RA_avg': 0.85,
                                                                                       'gender_gap': 0.18},
                                                     'two_person_images_diff_gender': {'RA_avg': 0.55,
                                                                                       'gender_gap': 0.49}}},
                                {'metadata': {'experiment_desc': 'CAPTIONING', 'model_name': 'instruct_blipv2'},
                                 'resolution_bias': {'all_images': {'overall_accuracy': 0.82},
                                                     'single_person_images': {'RA_avg': 0.94, 'gender_gap': 0.11},
                                                     'two_person_images': {'RA_avg': 0.7, 'gender_gap': 0.29},
                                                     'two_person_images_same_gender': {'RA_avg': 0.86,
                                                                                       'gender_gap': 0.15},
                                                     'two_person_images_diff_gender': {'RA_avg': 0.54,
                                                                                       'gender_gap': 0.42}}}]}

    overall_acc = []
    single_person_gaps = []
    two_person_gaps = []

    models = ['qwen', 'instruct_blipv2', 'llava1.5']
    for model in models:
        acc_values = []
        single_gap_values = []
        two_gap_values = []
        for item in data[model]:
            acc_values.append(item['resolution_bias']['all_images']['overall_accuracy'])
            single_gap_values.append(item['resolution_bias']['single_person_images']['gender_gap'])
            two_gap_values.append(item['resolution_bias']['two_person_images']['gender_gap'])
        overall_acc.append(acc_values)
        single_person_gaps.append(single_gap_values)
        two_person_gaps.append(two_gap_values)

    print("overall_acc =", overall_acc)
    print("single_person_gaps =", single_person_gaps)
    print("two_person_gaps =", two_person_gaps)



# 数据
labels = ['original', 'DR.GAP$_{manual}$', 'DR.GAP']
models = ['Qwen2-VL', 'InstructBLIP', 'Llava-1.5']  # 调整模型顺序
overall_acc = [
    [0.40, 0.34, 0.82],  # qwen
    [0.78, 0.80, 0.82],  # instruct_blipv2
    [0.72, 0.79, 0.82]  # llava1.5
]

single_person_gaps = [
    [0.59, 0.42, 0.1],   # qwen
    [0.35, 0.21, 0.11],  # instruct_blipv2
    [0.15, 0.13, 0.09]   # llava1.5
]
two_person_gaps = [
    [0.3, 0.19, 0.05],   # qwen
    [0.72, 0.5, 0.34],   # instruct_blipv2
    [0.73, 0.67, 0.65]   # llava1.5
]

overall_acc = [[0.4, 0.68, 0.82], [0.72, 0.76, 0.82], [0.78, 0.8, 0.82]]
single_person_gaps = [[0.59, 0.27, 0.1], [0.35, 0.24, 0.11], [0.15, 0.14, 0.09]]
two_person_gaps = [[0.3, 0.05, 0.05], [0.72, 0.34, 0.29], [0.73, 0.67, 0.65]]
resolution_bias = np.mean(np.array([single_person_gaps, two_person_gaps]), axis=0).tolist()

# 设置条形图的位置和宽度
bar_width = 0.15  # 调整条形宽度为更窄
index = np.arange(len(models)) + bar_width  # 调整索引位置以避免重叠

plt.rcParams['figure.dpi'] = 300
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))

# 定义颜色
colors = ["#F9E4D4", "#F5BF6F", "#C8E2E3"]
colors = ["#C6DCB9", "#F5BF6F", "#CEBAF0"]
# colors = ["#F3D8F1", "#FBDF9D", "#CDC6FF"]
# 绘制单人图像的gender_gap
for i in range(len(overall_acc[0])):
    ax1.bar(index + i * bar_width, [gap[i] for gap in overall_acc], bar_width, label=labels[i], color=colors[i])

ax1.set_title('Resolution Accuracy for VisoGender',fontsize=10,fontweight='bold')
# ax1.set_ylabel('Gender Gap')
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 1)

# 绘制双人图像的gender_gap
for i in range(len(resolution_bias[0])):
    ax2.bar(index + i * bar_width, [gap[i] for gap in resolution_bias], bar_width, label=labels[i], color=colors[i])

ax2.set_title('Resolution Bias for VisoGender',fontsize=10,fontweight='bold')
# ax2.set_ylabel('Gender Gap')
ax2.set_xticks(index + bar_width)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 1)

# 创建全局图例
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.7), fontsize=10)

# 显示图形
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # 调整底部间距以容纳图例
# plt.show()
plt.savefig(f'resimgs/visogender_vlms.jpg')

from PIL import Image

image = Image.open(f'resimgs/visogender_vlms.jpg')
rect = (0, 0, 1500, 1050)
crop_image = image.crop(rect)
crop_image.save(f'resimgs/visogender_vlms.jpg')