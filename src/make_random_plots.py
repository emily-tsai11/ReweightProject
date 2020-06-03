import time
import lheparser as lhe
import json
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as colors

start = time.time()

# print('parsing events into dict...')
# events = lhe.lhe_to_dict()

# print('writing events to json file...')
# with open('events_test.json', 'w') as f:
    # json.dump(events, f)

# print('writing events to root file...')
# lhe.dict_to_root(events, 'test')

with open('../data/events_test.json', 'r') as f:
    events = json.load(f)

couplings = [['M14K010', 'M14K015', 'M14K020', 'M14K025', 'M14K030', 'M14K035', 'M14K040', 'M14K045', 'M14K050', 'M14K060', 'M14K070', 'M14K080', 'M14K090', 'M14K100', 'M14K110', 'M14K120', 'M14K130', 'M14K140', 'M14K150', 'M14K160'], ['M15K010', 'M15K015', 'M15K020', 'M15K025', 'M15K030', 'M15K035', 'M15K040', 'M15K045', 'M15K050', 'M15K060', 'M15K070', 'M15K080', 'M15K090', 'M15K100', 'M15K110', 'M15K120', 'M15K130', 'M15K140', 'M15K150', 'M15K160']]


# find variables that affect weights
def wts_param(fignum, particle_num, param, label, xlabel):
    for mass in couplings:
        x = np.arange(len(mass))
        y = []
        labels = []
        for i in range(7): # first 7 events
            temp_y = []
            for j in range(len(mass)):
                temp_y.append(events[str(i)]['wts'][mass[j]])
            y.append(temp_y)
            labels.append(events[str(i)][str(particle_num)][param])

        plt.figure(fignum * len(couplings) + couplings.index(mass))
        for i in range(len(y)):
            plt.plot(x, y[i], linewidth = 0.9, label = label % labels[i])
        plt.grid()
        plt.legend(loc = 'best', fontsize = 10)
        plt.xticks(x, mass, fontsize = 7, rotation = 'vertical')
        plt.title(xlabel)
        plt.ylabel('Weight')
        plt.axis([-0.5, 19.5, 0, 0.3])
        plt.savefig('../plots/wts_param/wts_' + param + '_' + str(mass[0][1:3]) + '.png')

fignum = 0
wts_param(fignum, 2, 'm', 'Invariant M$_T$ = %.0f GeV', 'Weight vs. M$_{T}$')
fignum += 1
wts_param(fignum, 2, 'pz', 'p$_z$ = %.0f', 'Weight vs. p$_z$')
fignum += 1

# find how pt affects weights
for mass in couplings:
    x = np.arange(len(mass))
    y = []
    labels = []
    for j in range(10):
        event = events[str(j)]
        particle = event['2']
        _4vector = ROOT.TLorentzVector()
        _4vector.SetPxPyPzE(particle['px'], particle['py'], particle['pz'], particle['e'])

        temp_y = []
        for i in range(len(mass)):
            temp_y.append(events[str(j)]['wts'][mass[i]])
        y.append(temp_y)
        labels.append(_4vector.Pt())
        print(particle['m'], _4vector.Pt())

    plt.figure(fignum * len(couplings) + couplings.index(mass))
    for i in range(len(y)):
        plt.plot(x, y[i], linewidth = 0.9, label = 'p$_T$ = %.0f' % labels[i])
    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.xticks(x, mass, fontsize = 7, rotation = 'vertical')
    plt.title('p$_T$')
    plt.ylabel('Weight')
    plt.axis([-0.5, 19.5, 0, 0.3])
    plt.savefig('../plots/wts_param/wts_pt_' + str(mass[0][1:3]) + '.png')
    fignum += 1


'''
# 2D histogram of weights
mat = []
for i in range(len(couplings)):
    data = []
    for j in range(len(events)):
        data.append(events[str(j)]['wts'][couplings[i]])
    mat.append(np.histogram(data, bins = len(couplings), range = (0.0, 0.4))[0])
cmap = plt.get_cmap('ocean')
cmap.set_bad(color = 'black', alpha = 1.0)
plt.figure(0)
plt.matshow(mat, norm = colors.LogNorm(vmin = 0.01, vmax = 50), cmap = cmap)
plt.savefig('heatmap_weights.png')
plt.close(0)
'''

'''
# sum of weights plot (something about cross section)
data = []
for i in range(len(events)):
    sum = 0
    for j in range(len(couplings)):
        sum += events[str(i)]['wts'][couplings[j]]
    data.append(sum)
plt.figure(0)
plt.hist(data, 100)
plt.title('Frequency of Sum of Weights')
plt.xlabel('Sum of Weights')
plt.ylabel('Number of Occurrences')
plt.savefig('plot_SumWeights.png')
plt.close(0)
'''

'''
# make ordered ROOT file
ROOT.gROOT.SetBatch(True)

file = ROOT.TFile.Open('../data/hists_all.root', 'read')
new_file = ROOT.TFile.Open('../data/hists_T_m_pt.root', 'recreate')

for coupling in couplings:
    hist_name_m = 'm_T_2_rwt_' + coupling
    hist_m = file.Get(hist_name_m)
    hist_m.Write()
for coupling in couplings:
    hist_name_pt = 'pt_T_2_rwt_' + coupling
    hist_pt = file.Get(hist_name_pt)
    hist_pt.Write()

file.Close()
new_file.Close()
'''

print('RUNTIME: %.2f SECONDS.' % (time.time() - start))
