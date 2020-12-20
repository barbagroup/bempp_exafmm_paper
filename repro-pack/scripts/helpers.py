import numpy as np
import os
import bempp.api
import yaml

def parse_pqr(pqr_file):
    """
    Parse pqr file and return charges and their coordinates.
    """
    charges = list()
    coords = list()
    with open(pqr_file, 'r') as f: 
        for line in f:
            line = line.strip().split()
            if len(line) > 0:
                if line[0] == 'ATOM':
                    coords.append(line[-5:-2])
                    charges.append(line[-2])
    charges = np.array(charges, dtype=float)
    coords = np.array(coords, dtype=float).reshape(-1,3)

    return charges, coords

def generate_grid(face_file, vert_file):
    """
    Create bempp Grid object from .face and .vert files.
    """
    face = open(face_file, 'r').read()
    vert = open(vert_file, 'r').read()
    faces = np.vstack(np.char.split(face.split('\n')[0:-1]))[:,:3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split('\n')[0:-1]))[:,:3].astype(float)
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())

    return grid

def parse_config(config_file):
    """
    Parse config file and return parameters, grid, charges. 
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    mesh_dir = os.path.join(parent_dir, "mesh")
    pqr_dir = os.path.join(parent_dir, "pqr")

    with open(config_file) as f:
        global params
        params = yaml.load(f, Loader=yaml.FullLoader)
        ep_in = params['ep_in']
        ep_ex = params['ep_ex']
        kappa = params['kappa']
        bempp.api.GLOBAL_PARAMETERS.quadrature.regular = params['regular']
        bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = params['expansion_order']
        bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = params['ncrit']
        restart = params['restart']
        tol = params['tol']

        if 'refine_level' in params.keys():
            refine_level = int(params['refine_level'])
            grid = bempp.api.shapes.regular_sphere(refine_level)
        else:
            face_file = os.path.join(mesh_dir, params['face_file'])
            vert_file = os.path.join(mesh_dir, params['vert_file'])
            grid = generate_grid(face_file, vert_file)
        
        if 'ncharges' in params.keys():
            n_q = int(params['ncharges'])
            rand = np.random.RandomState(0)
            q = rand.rand(n_q)  # charges
            x_q = rand.rand(3*n_q) - 0.5
            x_q = x_q.reshape((n_q,3))
            x_q = 0.7 * x_q / np.linalg.norm(x_q, axis=1).reshape((n_q,1))
        else:
            pqr_file  = os.path.join(pqr_dir, params['pqr_file'])
            q, x_q = parse_pqr(pqr_file)

    return grid, q, x_q

