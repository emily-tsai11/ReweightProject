import xml.etree.ElementTree as ET
#import ROOT

def lhe_to_dict(lhefilename = 'unweighted_events.lhe'):
    events = {}
    eventindex = 0
    tree = ET.parse(lhefilename)
    root = tree.getroot()
    for ev in root.findall('event'):
        #print ev.text
        events[eventindex] = {}
        this_event = events[eventindex]
        eventinfo = ev.text.strip().split('\n')
        particleindex = 0
        for ii in range(len(eventinfo)):
            if ii == 0: continue
            this_event[particleindex] = {}
            this_particle = this_event[particleindex]
            particleinfo = eventinfo[ii].strip().split()
            for jj in range(len(particleinfo)):
                this_info = particleinfo[jj]
                if   jj == 0:  this_particle['pdgid'] = int(this_info)
                elif jj == 1:  this_particle['status'] = int(this_info)
                elif jj == 6:  this_particle['px'] = float(this_info)
                elif jj == 7:  this_particle['py'] = float(this_info)
                elif jj == 8:  this_particle['pz'] = float(this_info)
                elif jj == 9:  this_particle['e'] = float(this_info)
                elif jj == 10: this_particle['m'] = float(this_info)
                else: continue
            particleindex += 1
        this_event['wts'] = {'nominal': 1.0 }
        try:
            rwt = ev.find('rwgt')
            for wt in rwt.findall('wgt'):
                wtname = wt.get('id')
                wtval = float(wt.text.strip())
                this_event['wts'][wtname] = wtval
        except:
            pass
        eventindex += 1
        if eventindex == 50: break

    return events

def dict_to_root(evdict, tag):
    import ROOT
    pdgmap = {1:'d',
              2:'u',
              3:'s',
              4:'c',
              5:'b',
              6:'t',
              11:'e',
              13:'mu',
              15:'tau',
              12:'nu',
              14:'nu',
              16:'nu',
              21:'g',
              23:'Z',
              24:'W',
              25:'H',
              6000005:'X',
              6000006:'T',
              6000007:'B',
              6000008:'Y'}
    histset = { 'e0': ROOT.TH1D("e0", "incoming parton energy", 600, 0.0, 6000.0) ,
                'ht': ROOT.TH1D("ht", "sum of transverse momenta of final state hadrons and leptons", 600, 0.0, 6000.0),
            }
    for evkey in evdict.keys():
        event = evdict[evkey]
        for keywt in event['wts'].keys():
            ev_ht = 0.
            weight = event['wts'][keywt]
            particlekeys = event.keys()
            if keywt == 'nominal':
                wtsuffix = ''
            else:
                wtsuffix = '_rwt_' + keywt
                if 'ht' + wtsuffix not in histset.keys():
                    histset['ht' + wtsuffix] = ROOT.TH1D('ht' + wtsuffix, " ", 600, 0.0, 6000.0)
            for particlekey in particlekeys:
                if particlekey == 'wts':
                    continue
                particle = event[particlekey]
                pdgid = pdgmap[abs(particle['pdgid'])]
                status = particle['status']
                if particle['status'] == -1:
                    histkeys = ['e0' + wtsuffix]
                else:
                    histkeys = ['e_' + str(pdgid) + '_' + str(status) + wtsuffix,
                                'pt_' + str(pdgid) + '_' + str(status) + wtsuffix,
                                'm_' + str(pdgid) + '_' + str(status) + wtsuffix,
                                'eta_' + str(pdgid) + '_' + str(status) + wtsuffix,
                                'phi_' + str(pdgid) + '_' + str(status) + wtsuffix]
                _4vector = ROOT.TLorentzVector()
                _4vector.SetPxPyPzE(particle['px'], particle['py'], particle['pz'], particle['e'])
                for key in histkeys:
                    if key not in histset.keys():
                        if key[0:3] == 'eta': histset[key] = ROOT.TH1D(key, " ", 50, -5.0, 5.0)
                        elif key[0:3] == 'phi': histset[key] = ROOT.TH1D(key, " ", 32, -3.2, 3.2)
                        else: histset[key] = ROOT.TH1D(key, " ", 600, 0.0, 6000.0)
                    if key[0:2] == 'e_': histset[key].Fill(_4vector.E(), weight)
                    elif key[0:2] == 'pt': histset[key].Fill(_4vector.Pt(), weight)
                    elif key[0] == 'm': histset[key].Fill(_4vector.M(), weight)
                    elif key[0:3] == 'eta': histset[key].Fill(_4vector.Eta(), weight)
                    elif key[0:3] == 'phi': histset[key].Fill(_4vector.Phi(), weight)
                if status == 1: ev_ht += _4vector.Pt()

            histset['ht' + wtsuffix].Fill(ev_ht, weight)

        if evkey % 1000 == 0:
            print(evkey, " events filled in all histograms")

    f = ROOT.TFile('hists_' + tag + '.root', "RECREATE")
    f.cd()
    for histkey in histset.keys():
        histset[histkey].Write()
    f.Close()

#lhe_to_dict('/afs/cern.ch/work/a/avroy/MG_Reweight_testinATLAS2/base_run_test/tmp_run_01._00001.events.events')
