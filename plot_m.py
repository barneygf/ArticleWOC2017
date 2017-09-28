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
# c_prof = 5  # Number of column, in filename, with well log data
# c_lab = 8  # Number of column, in filename, with laboratory data
c_prof = 'PorowatoscCalkowitaPiknometria'  # Name of column, in filename, with well log data
c_lab = 'ParametrPorowatosci'  # Name of column, in filename, with laboratory data
x_axis_type = 'linear'  # Choose x axis scale - linear or log
y_axis_type = 'log'  # Choose y axis scale - linear or log
fig_size_x = 24
fig_size_y = 12
size_font = 12
rys_nr_x = 0.2
rys_nr_y = 16.5
empty_value = -999.25

variables = np.genfromtxt(fname=filename, dtype=str, delimiter=';', unpack=True, max_rows=1)
units = np.genfromtxt(fname=filename, dtype=str, skip_header=1, delimiter=';', unpack=True, max_rows=1)
data = pd.read_csv(filename, skiprows=2, delimiter=';', names=variables)
data = data[['Otwor', 'DEPTH', 'Litostratygrafia', c_prof, c_lab]]
data = data[(data[c_prof] != empty_value) & (data[c_lab] != empty_value)]  # Drop all rows with empty value
print(data)
data_lit = data.groupby('Litostratygrafia')
data_well = data.groupby('Otwor')
'''
for group, frame in data_gr:
    print(group)
    print(frame)
'''
#print(type(data['PorowatoscCalkowitaPiknometria'].iloc[5]))
# # Calculations:
# variables = np.genfromtxt(fname=filename, dtype=str, delimiter=';', unpack=True, max_rows=1)
# units = np.genfromtxt(fname=filename, dtype=str, skip_header=1, delimiter=';', unpack=True, max_rows=1)
# data = np.genfromtxt(fname=filename, dtype=str, skip_header=2, delimiter=';', unpack=True)
#
# data2 = {}
# for i in range(len(data)):
#     temp = data[i]
#     try:
#         temp = temp.astype(float)
#     except ValueError:
#         print('')
#     data2[variables[i]] = temp
#
# print(data2)
#
def cm2inch(*centy):  # Converting centimeters to inches
    inch = 2.54
    if isinstance(centy[0], tuple):
        return tuple(i/inch for i in centy[0])
    else:
        return tuple(i/inch for i in centy)

matplotlib.rc('xtick', labelsize=size_font)
matplotlib.rc('ytick', labelsize=size_font)

#f = plt.figure(figsize=cm2inch(fig_size_x, fig_size_y))  # Size of the figure in centimeters
#ax = fig.add_subplot(111)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=cm2inch(fig_size_x, fig_size_y))
point_size = 60

for group, frame in data_lit:
    print(group)
    print(frame)

#for i in data_lit[c_prof].get_group('Ng'):

# for i in range(len(data2[variables[0]])):
#     if data2[variables[c_prof]][i] != -999.25 and data2[variables[c_lab]][i] != -999.25:
#         if data2[variables[0]][i] == 'L-1':
#             ax1.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='r', label='A-1', s=point_size)
#         elif data2[variables[0]][i] == 'L-2':
#             ax1.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='dodgerblue', label='A-2',
#                        s=point_size, marker='^')
#         elif data2[variables[0]][i] == 'L-3':
#             ax1.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='g', label='A-3', s=point_size,
#                        marker='+')
#         elif data2[variables[0]][i] == 'L-4':
#             ax1.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='k', label='A-4', s=point_size,
#                        marker='x')
#
# # Calculating linear regression
# data3 = []
# data4 = []
# if filename == 'Cross_U.csv':
#     for i in range(len(data2[variables[0]])):
#         if data2[variables[c_prof]][i] < 4:
#             if data2[variables[c_prof]][i] != -999.25 and data2[variables[c_lab]][i] != -999.25:
#                 data3.append(data2[variables[c_prof]][i])
#                 data4.append(data2[variables[c_lab]][i])
#                 print(data3[-1], data4[-1])
# else:
#     for i in range(len(data2[variables[0]])):
#         print(i)
#         if data2[variables[c_prof]][i] != -999.25 and data2[variables[c_lab]][i] != -999.25:
#             data3.append(data2[variables[c_prof]][i])
#             data4.append(data2[variables[c_lab]][i])
#
# if x_axis_type == 'linear' and y_axis_type == 'linear':
#     slope, intercept, r_value, p_value, std_err = stats.linregress(data3, data4)
# elif x_axis_type == 'log' and y_axis_type == 'log':
#     slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(data3), np.log10(data4))
# elif x_axis_type == 'linear' and y_axis_type == 'log':
#     slope, intercept, r_value, p_value, std_err = stats.linregress(data3, data4)
#
# print('slope: ', slope)
# print('intercept: ', intercept)
# print('r_value: ', r_value)
#
# x_regr = np.linspace(np.min(data3), np.max(data3))
# y_regr = slope * x_regr + intercept
# #y_regr = 1074.6 * x_regr ** (-1.17)
#
#
# R2_text = r'y = 0.83x + 1.06''\n'r'$R^2 = $' + str('%.3f' % r_value**2)
# #R2_text = r'$P_p = 1074.6p^{-1.17}$''\n'r'$R^2 = 0.69$''\n'r'$m = 1.17$'
# ax1.plot(x_regr, y_regr, 'm--')
# if x_axis_type == 'linear' and y_axis_type == 'linear':
#     ax1.text(x_regr[-17], y_regr[-26] + 0.05 * y_regr[-5], R2_text, color='m',
#             fontsize=size_font)
# elif x_axis_type == 'log' and y_axis_type == 'log':
#     ax1.text(x_regr[-2], y_regr[-40] + 0.05 * y_regr[-40], R2_text, color='m',
#             fontsize=size_font)
# elif x_axis_type == 'linear' and y_axis_type == 'log':
#     ax1.text(x_regr[-25], y_regr[-5] + 0.05 * y_regr[-5], R2_text, color='m',
#             fontsize=size_font)
#
# ax1.text(rys_nr_x, rys_nr_y, 'a)', fontsize=size_font+8)
# ax2.text(rys_nr_x, rys_nr_y, 'b)', fontsize=size_font+8)
# handles, labels = ax1.get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# ax1.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc=2, fancybox=True, shadow=True, ncol=2,
#           fontsize=size_font)
# ax1.set_xlabel(variables[c_prof] + ' [' + units[c_prof] + ']', fontsize=size_font)
# ax1.set_ylabel(variables[c_lab] + ' Lab [' + units[c_lab] + ']', fontsize=size_font)
# ax1.set_xscale(x_axis_type)
# ax1.set_yscale(y_axis_type)
# ax1.set_axisbelow(True)
# ax1.grid()
# plt.tight_layout()
# ax1.set_xlim(-0.5, 22)
# ax1.set_ylim(-0.5, 22)
# #ax1.set_xlim(1, 100)
# #ax1.set_ylim(10, 10000)
# #plt.savefig(filename=variables[c_prof] + '-' + variables[c_lab] + '_Otw.png')
#
# #fig = plt.figure(figsize=cm2inch(fig_size_x, fig_size_y))  # Size of the figure in centimeters
# #ax = fig.add_subplot(111)
# #point_size = 40
# for i in range(len(data2[variables[0]])):
#     if data2[variables[c_prof]][i] != -999.25 and data2[variables[c_lab]][i] != -999.25:
#         if data2[variables[2]][i] == 'C1':
#             ax2.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='g', label='C1', s=point_size)
#         elif data2[variables[2]][i] == 'J2':
#             ax2.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='k', label='J2', s=point_size,
#                        marker='^')
#         elif data2[variables[2]][i] == 'J3':
#             ax2.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='dodgerblue', label='J3',
#                        s=point_size, marker='+')
#         elif data2[variables[2]][i] == 'Ng':
#             ax2.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='brown', label='Ng', s=point_size,
#                        marker='x')
#         elif data2[variables[2]][i] == 'T-P':
#             ax2.scatter(data2[variables[c_prof]][i], data2[variables[c_lab]][i], c='r', label='T-P', s=point_size,
#                        marker='d')
#
# ax2.plot(x_regr, y_regr, 'm--')
# if x_axis_type == 'linear' and y_axis_type == 'linear':
#     ax2.text(x_regr[-17], y_regr[-26] + 0.05 * y_regr[-5], R2_text, color='m',
#             fontsize=size_font)
# elif x_axis_type == 'log' and y_axis_type == 'log':
#     ax2.text(x_regr[-2], y_regr[-40] + 0.05 * y_regr[-40], R2_text, color='m',
#             fontsize=size_font)
# elif x_axis_type == 'linear' and y_axis_type == 'log':
#     ax2.text(x_regr[-25], y_regr[-5] + 0.05 * y_regr[-5], R2_text, color='m',
#             fontsize=size_font)
#
# ax2.text(70, 5000, 'b)', fontsize=size_font+8)
#
# handles, labels = ax2.get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# ax2.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc=2, fancybox=True, shadow=True, ncol=3,
#           fontsize=size_font)
# ax2.set_xlabel(variables[c_prof] + ' [' + units[c_prof] + ']', fontsize=size_font)
# #plt.ylabel(variables[c_lab] + ' Lab [' + units[c_lab] + ']', fontsize=size_font)
# ax2.set_xscale(x_axis_type)
# ax2.set_yscale(y_axis_type)
# ax2.set_axisbelow(True)
# ax2.grid()
# plt.tight_layout()
# ax2.set_xlim(-0.5, 22)
# ax2.set_ylim(-0.5, 22)
# #ax2.set_xlim(1, 100)
# #ax2.set_ylim(10, 10000)
# #plt.savefig(filename=variables[c_prof] + '-' + variables[c_lab] + '_Lit.png')
# plt.savefig(filename=variables[c_prof] + '-' + variables[c_lab] + '_razem.png')
# plt.show()