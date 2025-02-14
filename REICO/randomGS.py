#!/usr/bin/env python
#*********** v0.0.1 **************
# -*- coding: <encoding name> -*-

import numpy as np 
from ase.io import read,write
from ase import Atoms
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import json, sys
import time

def main():
    jdata = load_json(str(sys.argv[1]))
    if 'nproc' not in jdata.keys():jdata['nproc'] = 8
    if 'output' not in jdata.keys():jdata['output'] = 'random_structure.xyz'

    data = []
    for jobid,conf in jdata.items():
        if isinstance(conf,dict):
            argv = list(conf.keys())
            if 'mix' in argv:
                if 'vacuum' not in jdata[jobid].keys():
                    jdata[jobid]['vacuum'] = 0
                data.extend(mix_structure(mix=jdata[jobid]['mix'],numbers=jdata[jobid]['numbers'],\
                    vacuum=jdata[jobid]['vacuum'],nproc=jdata['nproc'],job_name=jobid))
            if 'ad' in argv:
                if 'vacuum' not in jdata[jobid].keys():
                    jdata[jobid]['vacuum'] = 8
                data.extend(adsorption_structure(ad=jdata[jobid]['ad'],numbers=jdata[jobid]['numbers'],\
                    vacuum=jdata[jobid]['vacuum'],nproc=jdata['nproc'],job_name=jobid))
            if 'cluster' in argv:
                if 'vacuum' not in jdata[jobid].keys():
                    jdata[jobid]['vacuum'] = 8
                data.extend(cluster_structure(cluster=jdata[jobid]['cluster'],numbers=jdata[jobid]['numbers'],\
                    vacuum=jdata[jobid]['vacuum'],nproc=jdata['nproc'],job_name=jobid))

    write(jdata['output'],data[:])

def load_json(json_file):
    filepath = Path(json_file)
    if filepath.suffix.endswith("json"):
        with Path(json_file).open() as fp:
            return json.load(fp)
    else:
        raise TypeError("config file is not json")

def atomic_radius(atom):
    radi = {
            'H':0.32,
            'Li':1.23,'Be':0.89,'B':0.82,'C':0.77,'N':0.7,'O':0.66,'F':0.64,
            'Na':1.54,'Mg':1.36,'Al':1.18,'Si':1.17,'P':1.1,'S':1.04,'Cl':0.99,
            'K':2.03,'Ca':1.74,'Sc':1.44,'Ti':1.32,'V':1.22,'Cr':1.18,'Mn':1.17,'Fe':1.17,'Co':1.16,'Ni':1.15,'Cu':1.17,'Zn':1.25,'Ga':1.26,'Ge':1.22,'As':1.21,'Se':1.17,'Br':1.14,
            'Rb':2.16,'Sr':1.91,'Y':1.62,'Zr':1.45,'Nb':1.34,'Mo':1.3,'Tc':1.27,'Ru':1.25,'Rh':1.25,'Pd':1.28,'Ag':1.34,'Cd':1.48,'In':1.44,'Sn':1.4,'Sb':1.41,'Te':1.37,'I':1.33,
            'Cs':2.35,'Ba':1.98,'Hf':1.44,'Ta':1.34,'W':1.3,'Re':1.28,'Os':1.26,'Ir':1.27,'Pt':1.3,'Au':1.34,'Hg':1.44,'Tl':1.48,'Pb':1.47,'Bi':1.46,'Po':1.46,'At':1.45,
            'La':1.88,'Ce':1.82,'Pr':1.83,'Nd':1.82,'Pm':1.81,'Sm':1.8,'Eu':2.04,'Gd':1.8,'Tb':1.78,'Dy':1.77,'Ho':1.77,'Er':1.76,'Tm':1.75,'Yb':1.94,'Lu':1.73,
            }

    return 1.3   #CHON等原子半径太小，随机生成的效率很低
    # return radi[atom]

def random_atoms_num(max_structure):
    random_structure, symbols, atoms_num = {}, '', 0

    for atom, num_range in max_structure.items():
        num_atom = np.random.RandomState().randint(num_range[0],num_range[1]+1)
        if num_atom>0:
            random_structure[atom] = num_atom
            symbols += f'{atom}{num_atom}'
            atoms_num += num_atom

    return random_structure, symbols, atoms_num

def get_volume(random_atoms):
    volume = 0
    for key,value in random_atoms.items():
        volume += 4/3 * np.pi * atomic_radius(key)**3 * value
    return volume

def random_cell(volume):
    min_length, max_length = (1.2*volume)**0.33, (5.0*volume)**0.34
    random_volume = 0
    while random_volume <= 1.2*volume or random_volume >= 5*volume:
        a,b,c = np.random.RandomState().uniform(min_length,max_length),\
                np.random.RandomState().uniform(min_length,max_length),\
                np.random.RandomState().uniform(min_length,max_length)
        # angle_a,angle_b,angle_c = np.random.RandomState().uniform(60,120),np.random.RandomState().uniform(60,120),np.random.RandomState().uniform(60,120)
        # random_volume = a*b*c*(1-np.cos(angle_a/180*np.pi)**2-np.cos(angle_b/180*np.pi)**2-np.cos(angle_c/180*np.pi)**2-2*np.cos(angle_a/180*np.pi)*np.cos(angle_b/180*np.pi)*np.cos(angle_c/180*np.pi))**0.5
        angle_a,angle_b,angle_c = 90, 90, 90
        random_volume = a*b*c
 
    return [a,b,c,angle_a,angle_b,angle_c]

def random_position(atoms):
    atom_distance = []
    positions = np.random.RandomState().uniform(0,1,[len(atoms),3])
    atoms.set_scaled_positions(positions)
    distance = atoms.get_all_distances(mic=True)
    for i in distance:
        atom_distance.append(sorted(i)[1])
    min_d,max_d = min(atom_distance), max(atom_distance)
    return atoms, min_d, max_d

def gen_structure(mix_structure,vacuum,tolerance_d=1.6):

    #start_time = time.time()
    #random_structure, symbols, atoms_num = random_atoms_num(mix_structure)
    #volume = get_volume(random_structure)    
    #cell = random_cell(volume)
    #atoms = Atoms(symbols = symbols,
    #                cell = cell,
    #                pbc = True)

    min_d, max_d = 0, 4
    while min_d <= tolerance_d or max_d >= 3.0:
        random_structure, symbols, atoms_num = random_atoms_num(mix_structure)
        volume = get_volume(random_structure)
        cell = random_cell(volume)
        atoms = Atoms(symbols = symbols,
                    cell = cell,
                    pbc = True)
        atoms,min_d,max_d = random_position(atoms)

    if vacuum:
        atoms.center(vacuum,axis=2)
    return atoms

def gen_adsorption(ad_structure,vacuum,tolerance_d):
    base_random_structure, base_symbols, base_atoms_num = \
            random_atoms_num(ad_structure['base'])
    ad_random_structure, ad_symbols, ad_atoms_num = \
            random_atoms_num(ad_structure['adsorption'])
    random_structure, symbols, atoms_num = \
            {}, base_symbols+ad_symbols, base_atoms_num+ad_atoms_num

    for key,value in base_random_structure.items():
        random_structure[key] = [value,value]

    for key,value in ad_random_structure.items():
        random_structure[key] = [value,value]

    atoms = 'restart'
    while atoms=='restart':
        atoms = gen_structure(random_structure, vacuum, tolerance_d)
    positions_z = sorted(atoms.positions,key=lambda pos:pos[2])
    base_pos, ad_pos = (np.array(positions_z[:base_atoms_num])).reshape(-1,3),\
                        (np.array(positions_z[base_atoms_num:])).reshape(-1,3)
    np.random.RandomState().shuffle(base_pos)
    np.random.RandomState().shuffle(ad_pos)
    positions = np.vstack([base_pos,ad_pos])

    atoms.set_positions(positions)

    return atoms

def mix_structure(mix, numbers, vacuum=0, tolerance_d=1.5, nproc=8, job_name='mix'):
    pbar = tqdm(total=numbers)
    pbar.set_description(job_name)
    update = lambda *args: pbar.update()
    p, results = Pool(nproc), []
    for i in range(numbers):
        results.append(p.apply_async(gen_structure,args=(mix, vacuum, tolerance_d), callback=update))

    p.close()
    p.join()
    gen_mix = [i.get() for i in results if i.get()]

    return gen_mix

def cluster_structure(cluster, numbers, vacuum=8, tolerance_d=1.5, nproc=8, job_name='cluster'):
    pbar = tqdm(total=numbers)
    pbar.set_description(job_name)
    update = lambda *args: pbar.update()
    p, results = Pool(nproc), []
    for i in range(numbers):
        results.append(p.apply_async(gen_structure,args=(cluster, 0, tolerance_d), callback=update))

    p.close()
    p.join()
    gen_clt = [i.get() for i in results if i.get()]
    for i in gen_clt:
        i.cell=i.get_cell_lengths_and_angles()[:3]
        i.center(vacuum,axis=[0,1,2])


    return gen_clt

def adsorption_structure(ad, numbers, vacuum=8, tolerance_d=1.5, nproc=8, job_name='ad'):
    pbar = tqdm(total=numbers)
    pbar.set_description(job_name)
    update = lambda *args: pbar.update()
    p, results = Pool(nproc), []
    for i in range(numbers):
        results.append(p.apply_async(gen_adsorption,args=(ad, vacuum, tolerance_d), callback=update))
    p.close()
    p.join()
    gen_ad = [i.get() for i in results if i.get()]

    return gen_ad

if __name__ == '__main__':
    main()
