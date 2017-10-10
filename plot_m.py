"""Maciej Barnaś, 2017-05-12
Plotting crossplots between well logs and laboratory data."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats
import pandas as pd

matplotlib.style.use('classic')

# Arguments to pass:
filename = 'Razem_Laboratorium.csv'
c_prof = 'WspPorCalkowitejNMR'  # Name of column, in filename, with well log data
unit_c_prof = '%'
c_lab = 'ParametrPorowatosci'  # Name of column, in filename, with laboratory data
unit_c_lab = '-'
x_axis_type = 'log'  # Choose x axis scale - linear or log
y_axis_type = 'log'  # Choose y axis scale - linear or log
fig_size_x = 24
fig_size_y = 12
size_font = 12
rys_nr_x = 1.1
rys_nr_y = 6000
empty_value = -999.25

variables = np.genfromtxt(fname=filename, dtype=str, delimiter=';', unpack=True, max_rows=1)
data = pd.read_csv(filename, skiprows=2, delimiter=';', names=variables)
data = data[['Otwor', 'DEPTH', 'Litostratygrafia', c_prof, c_lab]]
data = data[(data[c_prof] != empty_value) & (data[c_lab] != empty_value)]  # Drop all rows with empty value
#data = data[data[c_prof] > 1]
print(data)
data_lit = data.groupby('Litostratygrafia')
data_well = data.groupby('Otwor')

def cm2inch(*centy):  # Converting centimeters to inches
    inch = 2.54
    if isinstance(centy[0], tuple):
        return tuple(i/inch for i in centy[0])
    else:
        return tuple(i/inch for i in centy)

matplotlib.rc('xtick', labelsize=size_font)
matplotlib.rc('ytick', labelsize=size_font)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=cm2inch(fig_size_x, fig_size_y))
point_size = 60

# for group, frame in data_lit:
#     print(group)
#     print(frame)

colors_well = {'A-1': 'r', 'A-2': 'dodgerblue', 'A-3': 'g', 'A-4': 'k'}
markers_well = {'A-1': 'o', 'A-2': '^', 'A-3': '+', 'A-4': 'x'}
for group, frame in data_well:
    ax1.scatter(data_well[c_prof].get_group(group), data_well[c_lab].get_group(group), c=colors_well[group],
                label=group, s=point_size, marker=markers_well[group])

ax1.text(rys_nr_x, rys_nr_y, 'a)', fontsize=size_font+8)
ax2.text(rys_nr_x, rys_nr_y, 'b)', fontsize=size_font+8)
handles, labels = ax1.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc=1, fancybox=True, shadow=True, ncol=2,
          fontsize=size_font)
ax1.set_xlabel(c_prof + ' [' + unit_c_prof + ']', fontsize=size_font)
ax1.set_ylabel(c_lab + ' [' + unit_c_lab + ']', fontsize=size_font)
ax1.set_xscale(x_axis_type)
ax1.set_yscale(y_axis_type)
ax1.set_axisbelow(True)
ax1.grid()
ax1.set_xlim(10 ** 0, 10 ** 2)
ax1.set_ylim(10 ** 1, 10 ** 4)

# Calculating linear regression
if x_axis_type == 'linear' and y_axis_type == 'linear':
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[c_prof], data[c_lab])
elif x_axis_type == 'log' and y_axis_type == 'log':
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(data[c_prof]), np.log(data[c_lab]))
elif x_axis_type == 'linear' and y_axis_type == 'log':
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(data[c_prof]), np.log(data[c_lab]))

e_intercept = np.exp(intercept)
print('slope: ', slope)
print('intercept: ', intercept)
print('e_intercept', e_intercept)
print('r_value: ', r_value)
#
x_regr = np.linspace(0.8 * np.min(data[c_prof]), 1.4 * np.max(data[c_prof]))
y_regr = e_intercept * x_regr ** slope

R2_text = r'$P_p = %.1f$' % e_intercept + r'$p^{%.2f}$' % slope + '\n' + r'$R^2 = %.2f$' %abs(r_value) + '\n' + \
          r'$m = %.2f$' % abs(slope)
ax1.text(20, 110, R2_text, color='m', fontsize=size_font+1)
ax1.plot(x_regr, y_regr, 'm--')

colors_lit = {'C1': 'g', 'J2': 'k', 'J3': 'dodgerblue', 'Ng': 'brown', 'T-P': 'r'}
markers_lit = {'C1': 'o', 'J2': '^', 'J3': '+', 'Ng': 'x', 'T-P': 'd'}
for group, frame in data_lit:
    ax2.scatter(data_lit[c_prof].get_group(group), data_lit[c_lab].get_group(group), c=colors_lit[group],
                label=group, s=point_size, marker=markers_lit[group])

handles, labels = ax2.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc='best', fancybox=True, shadow=True, ncol=3,
           fontsize=size_font)

ax2.set_xlabel(c_prof + ' [' + unit_c_prof + ']', fontsize=size_font)
ax2.set_xscale(x_axis_type)
ax2.set_yscale(y_axis_type)
ax2.set_axisbelow(True)
ax2.grid()
ax2.set_xlim(10 ** 0, 10 ** 2 + 4)
ax2.set_ylim(10 ** 1, 10 ** 4)


# Calculating linear regression

# Define dictionaries:
slope2 = {}
intercept2 = {}
e_intercept2 = {}
r_value2 = {}
p_value2 = {}
std_err2 = {}

# Calculations
if x_axis_type == 'linear' and y_axis_type == 'linear':
    for group, frame in data_lit:
        slope2[group], intercept2[group], r_value2[group], p_value2[group], std_err2[group] = \
            stats.linregress(data_lit[c_prof].get_group(group), data_lit[c_lab].get_group(group))
        e_intercept2[group] = np.exp(intercept2[group])
elif x_axis_type == 'log' and y_axis_type == 'log':
    for group, frame in data_lit:
        slope2[group], intercept2[group], r_value2[group], p_value2[group], std_err2[group] = \
            stats.linregress(np.log(data_lit[c_prof].get_group(group)), np.log(data_lit[c_lab].get_group(group)))
        e_intercept2[group] = np.exp(intercept2[group])
elif x_axis_type == 'linear' and y_axis_type == 'log':
    for group, frame in data_lit:
        slope2[group], intercept2[group], r_value2[group], p_value2[group], std_err2[group] = \
            stats.linregress(np.log(data_lit[c_prof].get_group(group)), np.log(data_lit[c_lab].get_group(group)))
        e_intercept2[group] = np.exp(intercept2[group])

print('slope2: ', slope2)
print('intercept2: ', intercept2)
print('e_intercept2: ', e_intercept2)
print('r_value2: ', r_value2)

R2_text2 = {}
for group, frame in data_lit:
    R2_text2[group] = group + ': ' + r'$m = %.2f$' % abs(slope2[group]) + r', $R^2 = %.2f$' % abs(r_value2[group])

text_loc_lit = {'C1': (10, 800), 'J2': (10, 500), 'J3': (10, 300), 'Ng': (1.4, 20), 'T-P': (1.4, 12)}
for group, frame in data_lit:
    ax2.text(text_loc_lit[group][0], text_loc_lit[group][1], R2_text2[group], color=colors_lit[group],
             fontsize=size_font+1)

for group, frame in data_lit:
    #x_regr2 = [0.8 * np.min(data_lit[c_prof].get_group(group)), 1.4 * np.max(data_lit[c_prof].get_group(group))]
    x_regr2 = np.linspace(0.8 * np.min(data_lit[c_prof].get_group(group)),
                          1.4 * np.max(data_lit[c_prof].get_group(group)))
    y_regr2 = e_intercept2[group] * x_regr2 ** slope2[group]
    ax2.plot(x_regr2, y_regr2, color=colors_lit[group], linestyle='--')

plt.tight_layout()
plt.savefig(fname=c_prof + '-' + c_lab + '_m.png')
plt.show()

# writer = pd.ExcelWriter('output.xlsx')
# data.to_excel(writer,'Sheet1')
# writer.save()














